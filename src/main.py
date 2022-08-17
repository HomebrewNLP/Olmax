import copy
import math
import os
import time
import typing
import warnings

import jax
import jax._src.util as util
import wandb
import yaml
from jax import lax, numpy as jnp

from src.backend import device_id, loop
from src.constants import ParallelAxes
from src.context import Context, WhileTrainContext, init_class
from src.data import text_dataset
from src.model import body_ctx, compute
from src.optimizer import get_current_lr, update
from src.utils.checkpoint import read_ckpt, write_ckpt
from src.utils.wandblog import WandbLog


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0, True)
    data_slice = wctx.data[wctx.current_step % (wctx.ctx.training.device_steps * jax.process_count())]
    (loss, accuracy), grads = grad_fn(wctx.ctx.parameters, data_slice)
    update(wctx.ctx, grads, wctx.current_step)
    wctx.loss += loss
    wctx.top_loss += accuracy
    wctx.current_step += 1
    return wctx.serialize()


def jitless_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    training = wctx.ctx.training
    steps = training.device_steps * jax.process_count()
    step_batch, sequence_p1 = wctx.data.shape

    # "all-to-all" / "all-concat" with jax.process_count() outputs instead of jax.device_count() outputs
    # init sparse tensor with 0s everywhere except for local input slice
    data = jnp.zeros((jax.process_count(), step_batch, sequence_p1), wctx.data.dtype)
    data = data.at[device_id(wctx.ctx) // jax.process_count(), :, :].set(wctx.data)
    # same value was seen `local_device_count` times, so divide to remove implicit multiplication (int32 --> accurate)
    data = lax.psum(data, ParallelAxes.model).astype(wctx.data.dtype) // jax.local_device_count()

    # interleave samples within batch by transposing steps*process_count + batch and reshaping from (x,y).t() to x,y
    # process_count, steps * batch, sequence
    # --reshape--> batch, process_count * steps, sequence  ([[0, 1, 2], [3, 4, 5]]  -->  [[0, 1], [2, 3], [4, 5]])
    # --transpose--> process_count * steps, batch, sequence  ([[0, 1], [2, 3], [4, 5]] --> [[0, 2, 4], [1, 3, 5]])
    data = data.reshape(wctx.ctx.dims.batch, steps, sequence_p1).transpose(1, 0, 2)
    data = data[:1]
    wctx.data = jnp.stack([data[:, :, :-1], data[:, :, 1:]], 1)

    return loop(train_step, wctx.serialize(), 1, training.device_unroll)


def get_parameters(ctx: Context, inp: jnp.ndarray):
    def _fn(x: jnp.ndarray):
        initial_seed = ctx.seed
        initial_prng_key = ctx.prng_key
        ctx.seed += device_id(ctx)
        ctx.prng_key = jax.random.PRNGKey(ctx.seed)
        body_ctx(ctx, x)
        params = ctx.parameters
        var = ctx.parameter_variance
        ctx.parameters = {}
        ctx.prng_key = initial_prng_key
        ctx.seed = initial_seed
        ctx.parameter_variance = {}
        return params, jax.tree_util.tree_map(lambda x: lax.psum(x / ctx.dims.heads, ParallelAxes.model), var)

    inp = jnp.broadcast_to(inp, (len(jax.local_devices()),) + inp.shape)
    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=(0,), out_axes=(0, 0), donate_argnums=(0,))
    ctx.parameters, variance = pmapped(inp)
    ctx.parameter_variance = {name: var.mean() for name, var in variance.items()}


def get_optimizer_state(ctx: Context):
    def _fn(parameters: typing.Dict[str, jnp.ndarray]):
        new_ctx = ctx
        new_ctx.parameters = {}
        new_ctx = copy.deepcopy(new_ctx)
        new_ctx.parameters = parameters
        keys = jax.random.split(jax.random.PRNGKey(0), len(parameters))
        grads = {name: jax.random.truncated_normal(key, -2, 2, param.shape, ctx.model.computation_dtype) * 0.001
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


def train_loop(wctx: WhileTrainContext, step: typing.Callable):
    ctx = [wctx]

    def _fn(dat: jnp.ndarray) -> WhileTrainContext:
        w = ctx[0](dat)
        w.loss = jnp.zeros_like(w.loss)
        w.top_loss = jnp.zeros_like(w.loss)
        ctx[0] = WhileTrainContext(step(w.serialize()))
        return ctx[0]

    return _fn


def run_one(wblog: WandbLog):
    wctx = WhileTrainContext()
    wctx.ctx.is_initializing = True
    print(yaml.dump(wctx.ctx.config(), indent=4))
    device_steps = wctx.ctx.training.device_steps * jax.process_count()
    total_steps = wctx.ctx.training.steps * device_steps
    data = timeit("Initializing dataset", text_dataset, wctx.ctx)
    inp = timeit("Enqueueing first batch", next, data)[:wctx.ctx.dims.batch]
    timeit("Acquiring forward parameters", get_parameters, wctx.ctx, inp)
    parameter_count = sum(util.prod(param.shape) for name, param in wctx.ctx.parameters.items())
    timeit("Acquiring optimizer parameters", get_optimizer_state, wctx.ctx)
    buffer_count = sum(util.prod(param.shape) for name, param in wctx.ctx.parameters.items()) - parameter_count

    if wctx.ctx.training.checkpoint_load_path:
        read_ckpt(wctx.ctx)

    partition = {'parameters': {k: 0 for k in wctx.ctx.parameters.keys()},
                 'parameter_variance': {k: None for k in wctx.ctx.parameter_variance.keys()}, 'data': None,
                 'current_step': None, 'loss': None, 'top_loss': None
                 }

    wctx.current_step += wctx.ctx.training.start_step
    wblog.idx += wctx.ctx.training.start_step

    step = train_loop(wctx, timeit(f"PMapping across {ParallelAxes.model}", jax.pmap, jitless_step, ParallelAxes.model,
                                   in_axes=(partition,), out_axes=partition, donate_argnums=(0,)))

    timeit("Compiling model and performing first step", step, next(data))
    timeit("Running second step", step, next(data))
    print("\n")
    print(f"Parameters: {jax.process_count() * parameter_count:,}")
    print(f"Buffers:    {jax.process_count() * buffer_count:,}\n\n")

    start_time = time.time()
    for idx, dat in enumerate(data):
        step_start = time.time()
        wctx = step(dat)
        if idx % wctx.ctx.training.print_interval == 0:
            tokens_processed = device_steps * wctx.ctx.dims.sequence * wctx.ctx.dims.batch
            print(f'[{idx * device_steps:{len(str(total_steps))}d}/{total_steps}] '
                  f'Loss: {wctx.loss / device_steps:6.3f} - '
                  f'Accuracy: {wctx.top_loss / device_steps:8.3f} | '
                  f'LearningRate: {float(get_current_lr(wctx.ctx, wctx.current_step)):.5f} | '
                  f'StepTime: {time.time() - step_start:10.6f}s - '
                  f'Rate: {tokens_processed * (idx + 1) / (time.time() - start_time):9,.1f} Tokens/s')
        if jnp.isnan(wctx.loss):
            print("Loss is NaN")
            return wblog.loss_medians[-1]
        if wctx.ctx.wandb.use_wandb and idx % wctx.ctx.wandb.log_frequency == 0:
            if wblog(wctx, get_current_lr(wctx.ctx, wctx.current_step)):
                pass  # return wblog.loss_medians[-1]
        log_step = math.log2((idx + 1) * device_steps + 1)
        el = wctx.ctx.training.early_stopping.expected_loss
        expected_loss = el.offset + el.scale * math.exp(el.exponent * log_step)
        patience = 1 + wctx.ctx.training.early_stopping.loss_patience ** log_step
        threshold = patience * expected_loss
        if wblog.loss_medians[-1] > threshold:
            print(f"Worse than threshold | Current Median: {wblog.loss_medians[-1]:9.6f} - Threshold: {threshold:4.1f}")
            return wblog.loss_medians[-1]
        if wctx.ctx.training.trace.do_trace:
            if idx == wctx.ctx.training.trace.start_step:
                jax.profiler.start_trace(wctx.ctx.training.trace.output_path)
            if idx == wctx.ctx.training.trace.stop_step:
                jax.profiler.stop_trace()
        if wctx.ctx.training.do_checkpoint and (idx + 1) % (wctx.ctx.training.checkpoint_interval // device_steps) == 0:
            write_ckpt(wctx.ctx)


def dump_ctx(ctx: Context, run):
    with open("config.yaml", 'w') as f:
        f.write(yaml.dump(ctx.config(), indent=4))
    os.environ['CONFIG'] = 'config.yaml'
    run.config.update(ctx.config(), allow_val_change=True)


def main():
    warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")
    warnings.filterwarnings("ignore", message=".*Some donated buffers were not usable.*")
    # jax.config.update("jax_disable_jit", True)
    wctx = WhileTrainContext()
    ctx = wctx.ctx

    run = wandb.init(project=ctx.wandb.project, entity=ctx.wandb.entity, config=ctx.config(), name=ctx.wandb.name,
                     id=ctx.wandb.id)

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

    wblog = WandbLog(run, ctx.training.device_steps * jax.process_count())
    return run_one(wblog)


if __name__ == '__main__':
    main()
