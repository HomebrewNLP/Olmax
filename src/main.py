import copy
import sys
import time
import typing
import warnings

import jax
import jax._src.util as util
import optuna
import wandb
import yaml
from jax import numpy as jnp

from src.backend import loop
from src.constants import ParallelAxes
from src.context import Context, WhileTrainContext
from src.data import text_dataset
from src.model import compute, body_ctx
from src.optimizer import get_current_lr, update
from src.utils.checkpoint import write_ckpt
from src.utils.wandb import WandbLog


def train_step(inp: typing.Tuple[typing.Dict[str, typing.Any], jnp.ndarray]) -> typing.Dict[str, typing.Any]:
    while_ctx_dict, device_idx = inp
    wctx = WhileTrainContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0, True)
    (loss, accuracy), grads = grad_fn(wctx.ctx.parameters,
                                      wctx.data[wctx.current_step % wctx.ctx.training.device_steps],
                                      device_idx)
    update(wctx.ctx, grads, wctx.current_step)
    wctx.loss += loss
    wctx.top_loss += accuracy
    wctx.current_step += 1
    return wctx.serialize()


def jitless_step(while_ctx_dict: typing.Dict[str, typing.Any], device_idx:jnp.ndarray) -> typing.Dict[str, typing.Any]:
    training = WhileTrainContext(while_ctx_dict).ctx.training

    return loop(train_step, (while_ctx_dict, device_idx), training.device_steps, training.device_unroll)


def get_parameters(ctx: Context, inp: jnp.ndarray):
    def _fn(x: jnp.ndarray, idx: jnp.ndarray):
        initial_seed = ctx.seed
        initial_prng_key = ctx.prng_key
        ctx.seed += idx
        ctx.prng_key = jax.random.PRNGKey(ctx.seed)
        body_ctx(ctx, x)
        params = ctx.parameters
        var = ctx.parameter_variance
        ctx.parameters = {}
        ctx.prng_key = initial_prng_key
        ctx.seed = initial_seed
        ctx.parameter_variance = {}
        return params, var

    inp = jnp.broadcast_to(inp, (ctx.dims.sizes.heads,) + inp.shape)
    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=(0, 0), out_axes=(0, 0))
    ctx.parameters, variance = pmapped(inp, jnp.arange(ctx.dims.sizes.heads))
    ctx.parameter_variance = {name: var.mean() for name, var in variance.items()}


def get_optimizer_state(ctx: Context):
    def _fn(parameters: typing.Dict[str, jnp.ndarray]):
        new_ctx = ctx
        new_ctx.parameters = {}
        new_ctx = copy.deepcopy(new_ctx)
        new_ctx.parameters = parameters
        grads = {name: jnp.zeros_like(param) for name, param in parameters.items()}
        update(new_ctx, grads, jnp.ones((), dtype=new_ctx.model.computation_dtype))
        return new_ctx.parameters

    pmapped = jax.pmap(_fn, ParallelAxes.model, in_axes=({k: 0 for k in ctx.parameters.keys()},), out_axes=0)
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
        w.current_step = w.current_step + 1
        ctx[0] = WhileTrainContext(step(w.serialize()))
        return ctx[0]

    return _fn


def run_one(wblog: typing.Optional[WandbLog] = None, trial: typing.Optional[optuna.Trial] = None):
    wctx = WhileTrainContext()
    ctx = wctx.ctx
    ctx.is_initializing = True
    storage = ctx.wandb.storage
    ctx.wandb.storage = ""  # mask to avoid logging it publicly
    print(yaml.dump(ctx.config(), indent=4))
    ctx.wandb.storage = storage
    total_steps = ctx.training.steps * ctx.training.device_steps
    data = timeit("Initializing dataset", text_dataset, ctx)
    inp = timeit("Enqueueing first batch", next, data)[0, 0]
    timeit("Acquiring forward parameters", get_parameters, ctx, inp)
    parameter_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items())
    timeit("Acquiring optimizer parameters", get_optimizer_state, ctx)
    buffer_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items()) - parameter_count

    ctx.parameter_dims = {name: [ctx.dims.heads] + dims for name, dims in ctx.parameter_dims.items()}
    # It's not used anywhere, but nice to have

    partition = {'parameters': {k: 0 for k in ctx.parameters.keys()},
                 'parameter_variance': {k: None for k in ctx.parameter_variance.keys()}, 'data': None,
                 'current_step': None, 'loss': None, 'top_loss': None}
    step = train_loop(wctx, timeit(f"PMapping across {ParallelAxes.model}", jax.pmap, jitless_step, ParallelAxes.model,
                                   in_axes=(partition, 0), out_axes=partition))

    timeit("Compiling model and performing first step", step, next(data), jnp.arange(ctx.dims.sizes.heads))
    timeit("Running second step", step, next(data), jnp.arange(ctx.dims.sizes.heads))
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
        if jnp.isnan(wctx.loss) or wctx.top_loss == 0:
            return wblog.loss_medians[-1]
        if ctx.wandb.use_wandb and idx % ctx.wandb.log_frequency == 0:
            if wblog(wctx, get_current_lr(wctx.ctx, wctx.current_step)):
                return wblog.loss_medians[-1]
        if trial is not None:
            trial.report(wblog.loss_medians[-1], idx * ctx.training.device_steps)
            if trial.should_prune():
                return wblog.loss_medians[-1]
        thres = min((v for k, v in ctx.training.loss_thresholds.items() if idx * ctx.training.device_steps > k),
                    default=10 ** 9)
        if wblog.loss_medians[-1] > thres:
            return wblog.loss_medians[-1]
        if ctx.training.trace.do_trace:
            if idx == ctx.training.trace.start_step:
                jax.profiler.start_trace(ctx.training.trace.output_path)
            if idx == ctx.training.trace.stop_step:
                jax.profiler.stop_trace()
        if ctx.training.do_checkpoint and (idx + 1) % ctx.training.checkpoint_interval == 0:
            write_ckpt(ctx)


def main():
    warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")
    # jax.config.update("jax_disable_jit", True)
    wctx = WhileTrainContext()
    ctx = wctx.ctx
    if not ctx.wandb.use_wandb:
        return run_one()

    run = wandb.init(project=ctx.wandb.project, entity=ctx.wandb.entity, config=ctx.config())
    wblog = WandbLog(run)

    if "placeholder" not in run.config:
        return run_one(wblog)

    def objective(trial: optuna.trial.Trial) -> typing.Optional[float]:
        ctx.optimizer.exponential_decay = trial.suggest_float("exponential_decay", 1e-6, 1e-3, log=True)
        ctx.optimizer.weight_decay = trial.suggest_float("weight_decay", 1e-5, 1, log=True)
        ctx.optimizer.adam_beta1 = trial.suggest_float("adam_beta1", 1e-3, 1, log=True)
        ctx.optimizer.adam_beta2 = trial.suggest_float("adam_beta2", 1e-4, 1, log=True)
        ctx.optimizer.gradient_clip = trial.suggest_float("gradient_clip", 1e-5, 1, log=True)
        ctx.optimizer.learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
        ctx.optimizer.warmup_end = trial.suggest_int("warmup_end", 16, 4096, log=True)
        ctx.optimizer.momentum_beta = trial.suggest_float("momentum_beta", 1e-4, 1, log=True)
        ctx.dims.sizes.full_conv_kernel = trial.suggest_int("full_conv_kernel", 2, 16, log=True)
        ctx.dims.sizes.depthwise_conv_kernel = trial.suggest_int("depthwise_conv_kernel", 4, 256, log=True)
        ctx.dims.sizes.depth = trial.suggest_int("depth", 1, 16, log=True)
        ctx.dims.sizes.features_per_head = 128 * trial.suggest_int("features_per_head//128", 1, 3, log=True)
        ctx.dims.sizes.batch = 2 * trial.suggest_int("batch//2", 1, 8, log=True)
        ctx.model.rezero_lr_scale = trial.suggest_float("rezero_lr_scale", 1e-3, 2, log=True)
        ctx.dims.sizes.intermediate = ctx.dims.sizes.features_per_head
        ctx.dims.sizes.intermediate *= trial.suggest_int("group_linear_factor", 1, 4, log=True)
        ctx.model.leaky_relu_slope = trial.suggest_float("leaky_relu_slope", 1e-3, 2, log=True)
        ctx.model.glu_mode = trial.suggest_int("glu_mode", 0, 3)
        ctx.training.z_loss = trial.suggest_float("z_loss", 1e-3, 2, log=True)

        with open("config.yaml", 'w') as f:
            f.write(yaml.dump(ctx.config(), indent=4))
        sys.argv.insert(1, "config.yaml")
        ctx.wandb.storage = ""
        run.config.update(ctx.config(), allow_val_change=True)
        return run_one(wblog, trial)


    study = optuna.load_study(ctx.wandb.entity, ctx.wandb.storage, optuna.samplers.TPESampler(n_startup_trials=128),
                              optuna.pruners.PercentilePruner(ctx.wandb.percentile, n_startup_trials=128,
                                                              n_warmup_steps=2048))
    return study.optimize(objective, 1)


if __name__ == '__main__':
    main()
