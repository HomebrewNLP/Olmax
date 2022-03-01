import time
import typing
import warnings

import jax
import jax._src.util as util
import wandb
import yaml
from jax import numpy as jnp

from src.backend import loop
from src.constants import ParallelAxes
from src.context import Context, WhileTrainContext
from src.data import text_dataset
from src.model import compute, body_ctx
from src.optimizer import get_current_lr, update
from src.utils.wandb import WandbLog


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0, True)
    (loss, accuracy), grads = grad_fn(wctx.ctx.parameters,
                                      wctx.data[wctx.current_step % wctx.ctx.training.device_steps])
    update(wctx.ctx, grads, wctx.current_step)
    wctx.loss += loss
    wctx.top_loss += accuracy
    wctx.current_step += 1
    return wctx.serialize()


def jitless_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    training = WhileTrainContext(while_ctx_dict).ctx.training
    return loop(train_step, while_ctx_dict, training.device_steps, training.device_unroll)


def get_parameters(ctx: Context, inp: jnp.ndarray):
    def _fn(x):
        body_ctx(ctx, x)
        params = ctx.parameters
        var = ctx.parameter_variance
        ctx.parameters = {}
        ctx.parameter_variance = {}
        return params, var

    inp = jnp.broadcast_to(inp, (ctx.dims.sizes.heads,) + inp.shape)
    ctx.parameters, variance = jax.pmap(_fn, ParallelAxes.model, in_axes=0, out_axes=(0, None))(inp)
    ctx.parameter_variance = {name: var.mean() for name, var in variance.items()}
    ctx.parameter_dims = {name: [ctx.dims.heads] + dims for name, dims in ctx.parameter_dims.items()}


def get_optimizer_state(ctx: Context):
    def _fn(parameters: typing.Dict[str, jnp.ndarray], grads: typing.Dict[str, jnp.ndarray]):
        ctx.parameters = parameters
        update(ctx, grads, jnp.ones((), dtype=ctx.model.computation_dtype))
        params = ctx.parameters
        ctx.parameters = {}
        return params

    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=(0, 0), out_axes=0)
    ctx.parameters = pmapped(ctx.parameters, {name: jnp.zeros_like(param) for name, param in ctx.parameters.items()})
    jnp.ones([], dtype=ctx.model.computation_dtype)


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
        w.current_step = w.current_step + 1
        ctx[0] = WhileTrainContext(step(w.serialize()))
        return ctx[0]

    return _fn


def main():
    warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")
    # jax.config.update("jax_disable_jit", True)
    wctx = WhileTrainContext()
    ctx = wctx.ctx
    print(yaml.dump(ctx.config(), indent=4))
    ctx.is_initializing = True
    if ctx.wandb.use_wandb:
        run = wandb.init(project=ctx.wandb.project, entity=ctx.wandb.entity, config=ctx.config())
        wblog = WandbLog(run)
    total_steps = ctx.training.steps * ctx.training.device_steps
    data = timeit("Initializing dataset", text_dataset, ctx)
    inp = timeit("Enqueueing first batch", next, data)[0, 0]
    timeit("Acquiring forward parameters", get_parameters, ctx, inp)
    parameter_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items())
    timeit("Acquiring optimizer parameters", get_optimizer_state, ctx)
    buffer_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items()) - parameter_count

    partition = {'parameters': 0, 'data': None, 'current_step': None, 'loss': None, 'top_loss': None,
                 'parameter_variance': None}
    step = train_loop(wctx, timeit(f"PMapping across {ParallelAxes.model}", jax.pmap, jitless_step, ParallelAxes.model,
                                   in_axes=(partition,), out_axes=partition))

    global_start = time.time()

    timeit("Compiling model and performing first step", step, next(data))
    timeit("Running second step", step, next(data))
    print(f"\n\nParameters: {parameter_count:,}\nBuffers:    {buffer_count:,}\n\n")

    start_time = time.time()
    for idx, dat in enumerate(data):
        step_start = time.time()
        wctx = step(dat)
        if idx % ctx.training.print_interval == 0:
            millions_processed = ctx.training.device_steps * ctx.dims.sizes.sequence * ctx.dims.sizes.batch
            print(f'[{idx * ctx.training.device_steps:{len(str(total_steps))}d}/{total_steps}] '
                  f'Loss: {wctx.loss / ctx.training.device_steps:6.3f} - '
                  f'TopLoss: {wctx.top_loss / ctx.training.device_steps:8.3f} | '
                  f'LearningRate: {float(get_current_lr(ctx, wctx.current_step)):.5f} | '
                  f'StepTime: {time.time() - step_start:10.6f}s - '
                  f'Rate: {millions_processed * (idx + 1) / (time.time() - start_time):9,.1f} Tokens/s')
        if ctx.wandb.use_wandb and idx % ctx.wandb.log_frequency == 0:
            wblog(wctx, get_current_lr(wctx.ctx, wctx.current_step))
        if ctx.training.trace.do_trace:
            if idx == ctx.training.trace.start_step:
                jax.profiler.start_trace(ctx.training.trace.output_path)
            if idx == ctx.training.trace.stop_step:
                jax.profiler.stop_trace()


if __name__ == '__main__':
    main()
