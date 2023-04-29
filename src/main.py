import copy
import math
import os
import time
import warnings
from typing import Tuple, Dict, Any, Callable, Iterator

import jax
import numpy as np
import wandb
from jax import lax, numpy as jnp

from src.backend import deep_replace, device_id, loop
from src.constants import ParallelAxes
from src.context import Context, WhileTrainContext, init_class
from src.data import text_dataset
from src.model.main import body_ctx, compute
from src.optimizer import get_current_lr, update
from src.utils.checkpoint import read_train_checkpoint, write_train_checkpoint
from src.utils.wandblog import WandbLog



def train_step(while_ctx_dict: Dict[str, Any]) -> Dict[str, Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    steps = wctx.ctx.training.device_steps * jax.process_count()
    grad_fn = jax.value_and_grad(compute, 0, True)
    data_slice = wctx.data[wctx.current_step % steps]
    params = {k: v for k, v in wctx.ctx.parameters.items() if '/optimizer' not in k}
    scalars, grads = grad_fn(params, data_slice)
    update(wctx.ctx, grads, wctx.current_step)
    wctx.scalars += scalars / steps  # higher numerical accuracy if we divide before summing
    wctx.current_step += 1
    return wctx.serialize()


def jitless_step(while_ctx_dict: Dict[str, Any]) -> Dict[str, Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    training = wctx.ctx.training
    steps = training.device_steps * jax.process_count()
    step_batch, sequence_p1 = wctx.data.shape

    # "all-to-all" / "all-concat" with jax.process_count() outputs instead of jax.device_count() outputs
    # init sparse tensor with 0s everywhere except for local input slice
    data = jnp.zeros((jax.process_count(), step_batch, sequence_p1), wctx.data.dtype)
    data = data.at[jax.process_index(), :, :].set(wctx.data)
    # same value was seen `local_device_count` times, so divide to remove implicit multiplication (int32 --> accurate)
    data = lax.psum(data, ParallelAxes.model).astype(wctx.data.dtype) // jax.local_device_count()

    # interleave samples within batch by transposing steps*process_count + batch and reshaping from (x,y).t() to x,y
    # process_count, steps * batch, sequence
    # --reshape--> batch, process_count * steps, sequence  ([[0, 1, 2], [3, 4, 5]]  -->  [[0, 1], [2, 3], [4, 5]])
    # --transpose--> process_count * steps, batch, sequence  ([[0, 1], [2, 3], [4, 5]] --> [[0, 2, 4], [1, 3, 5]])
    data = data.reshape(wctx.ctx.dims.batch, steps, sequence_p1).transpose(1, 0, 2)
    wctx.data = jnp.stack([data[:, :, :-1], data[:, :, 1:]], 1)

    return loop(train_step, wctx.serialize(), steps, training.device_unroll)


def get_parameters(ctx: Context, inp: jax.Array):
    def _fn(x: jax.Array):
        initial_seed = ctx.seed
        initial_prng_key = ctx.prng_key
        ctx.seed += device_id()
        ctx.prng_key = jax.random.PRNGKey(ctx.seed)
        body_ctx(ctx, x)
        params = ctx.parameters
        var = ctx.parameter_variance
        ctx.parameters = {}
        ctx.prng_key = initial_prng_key
        ctx.seed = initial_seed
        ctx.parameter_variance = {}
        return params, lax.pmean(var, ParallelAxes.model)

    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=(0,), out_axes=(0, 0), donate_argnums=(0,))
    ctx.parameters, variance = pmapped(inp)
    ctx.parameter_variance = {name: var.mean() for name, var in variance.items()}


def get_optimizer_state(ctx: Context):
    def _fn(parameters: Dict[str, jax.Array]):
        new_ctx = ctx
        new_ctx.parameters = {}
        new_ctx = copy.deepcopy(new_ctx)
        new_ctx.parameters = parameters.copy()
        keys = jax.random.split(jax.random.PRNGKey(0), len(parameters))
        grads = {name: jax.random.uniform(key, param.shape, ctx.model.computation_dtype, 1e-6, 1e-3)
                 for key, (name, param) in zip(keys, parameters.items())}
        update(new_ctx, grads, jnp.ones((), dtype=new_ctx.model.computation_dtype))
        return new_ctx.parameters

    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=({k: 0 for k in ctx.parameters.keys()},), out_axes=0,
                       donate_argnums=(0,))
    ctx.parameters = pmapped(ctx.parameters)


def timeit(text: str, fn, *args, pad=50, **kwargs):
    start_time = time.time()
    print(f'{text}..', end='', flush=True)
    out = fn(*args, **kwargs)
    print(f"{' ' * (pad - len(text))}Took:{time.time() - start_time:9.2f}s", flush=True)
    return out


class TrainLoop:
    def __init__(self, wctx: WhileTrainContext, step: Callable):
        self.wctx = wctx
        self.step = step

    def __call__(self, dat: jax.Array) -> WhileTrainContext:
        wctx = self.wctx(dat)
        wctx.scalars = jnp.zeros_like(wctx.scalars)
        self.wctx = WhileTrainContext(self.step(wctx.serialize()))
        return self.wctx


def replicate(x: Any) -> Any:
    return jax.device_put_replicated(x, jax.local_devices())


def init_data(ctx: Context, skipped_samples: int) -> Tuple[Iterator[np.ndarray], np.ndarray]:
    np_data = timeit("Initializing dataset", text_dataset, ctx, skipped_samples)

    data = map(replicate, np_data)
    inp = timeit("Enqueueing first batch", next, data)[:, :ctx.dims.batch, :ctx.dims.sequence]
    return data, inp


def init_data_and_model(wctx: WhileTrainContext) -> Iterator[np.ndarray]:
    """Model gets loaded in-place into the `WhileTrainContext`"""
    if wctx.ctx.training.checkpoint_load_path:
        read_train_checkpoint(wctx, '[0]{100}')
        skipped_samples = math.ceil(wctx.step / jax.process_count() / wctx.ctx.training.device_steps)
        data, _ = init_data(wctx.ctx, skipped_samples)
        return data

    data, inp = init_data(wctx.ctx, 0)
    wctx.ctx.is_initializing = True
    timeit("Acquiring forward parameters", get_parameters, wctx.ctx, inp)
    timeit("Acquiring optimizer parameters", get_optimizer_state, wctx.ctx)
    wctx.ctx.is_initializing = False
    wctx.ctx.parameter_variance = replicate(wctx.ctx.parameter_variance)
    wctx.current_step = replicate(wctx.current_step)
    wctx.scalars = replicate(wctx.scalars)

    return data


def dump_ctx(ctx: Context, run):
    with open("config.yaml", 'w') as f:
        f.write(str(ctx))
    os.environ['CONFIG'] = 'config.yaml'
    run.config.update(ctx.config(), allow_val_change=True)


def main():
    warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")
    warnings.filterwarnings("ignore", message=".*Some donated buffers were not usable.*")

    wctx = WhileTrainContext()
    ctx = wctx.ctx

    run = wandb.init(project=ctx.wandb.project, entity=ctx.wandb.entity, config=ctx.config(), name=ctx.wandb.name,
                     id=ctx.wandb.id, group=ctx.wandb.group)

    cfg = {}
    for param_name, param in run.config.items():
        if '.' not in param_name:
            continue
        inner_cfg = cfg
        split_name = param_name.split(".")
        for s in split_name[:-1]:
            if s not in inner_cfg:
                inner_cfg[s] = {}
            inner_cfg = inner_cfg[s]
        inner_cfg[split_name[-1]] = param
    init_class(ctx, cfg)
    dump_ctx(ctx, run)

    wctx = WhileTrainContext()
    print(wctx.ctx)
    device_steps = wctx.ctx.training.device_steps * jax.process_count()
    total_steps = wctx.ctx.training.steps * device_steps
    tokens_processed = wctx.ctx.dims.sequence * wctx.ctx.dims.batch
    data = init_data_and_model(wctx)
    parameter_count = sum(param.size for name, param in wctx.ctx.parameters.items() if "optimizer" not in name)
    buffer_count = sum(param.size for name, param in wctx.ctx.parameters.items()) - parameter_count

    partition = deep_replace(wctx.serialize(), 0)

    step = jax.pmap(jitless_step, ParallelAxes.model, in_axes=(partition,), out_axes=partition, donate_argnums=(0,))
    step = TrainLoop(wctx, step)

    print("\n")
    print(f"Parameters: {jax.process_count() * parameter_count:,}")
    print(f"Buffers:    {jax.process_count() * buffer_count:,}\n\n")

    checkpoint_at = wctx.ctx.training.checkpoint_interval + wctx.step
    start_time = time.time()
    wblog = WandbLog(run, int(ctx.training.device_steps * jax.process_count()), parameter_count, tokens_processed)
    for idx, dat in enumerate(data):
        step_start = time.time()
        wctx = step(dat)
        current_step = int(wctx.step)
        lr = float(get_current_lr(wctx.ctx, wctx.current_step[0]))
        print(f'[{current_step:{len(str(total_steps))}d}/{total_steps}] '
              f'Loss: {wctx.scalars[0, 0]:6.3f} - '
              f'Accuracy: {wctx.scalars[0, 1]:8.3f} | '
              f'LearningRate: {lr:.5f} | '
              f'StepTime: {time.time() - step_start:10.6f}s - '
              f'Rate: {tokens_processed * (current_step + 1) / (time.time() - start_time):9,.1f} Tokens/s')
        if wblog(wctx, current_step, lr):
            return
        if wctx.ctx.training.trace.do_trace:
            if idx == wctx.ctx.training.trace.start_step:
                jax.profiler.start_trace(wctx.ctx.training.trace.output_path)
            if idx == wctx.ctx.training.trace.stop_step:
                jax.profiler.stop_trace()
        if wctx.ctx.training.do_checkpoint and current_step > checkpoint_at:
            write_train_checkpoint(wctx)
            checkpoint_at += wctx.ctx.training.checkpoint_interval
    return


if __name__ == '__main__':
    main()
