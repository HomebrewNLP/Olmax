import typing

import jax
from jax import lax, numpy as jnp

from .backend import assign, get_param, prefixed_name, stable_rsqrt, with_context, zero_param
from .context import Context
from .shampoo import Preconditioner, fallback_pth_root


def one_shape(ndim: int, dim_name: int, dim_idx: int) -> typing.List[int]:
    base = [1] * ndim
    base[dim_idx] = dim_name
    return base


@with_context(count=False)
def sm3(ctx: Context, grad: jnp.ndarray) -> jnp.ndarray:
    weight_update = zero_param(ctx, "dim0", one_shape(grad.ndim, grad.shape[0], 0), ctx.model.storage_dtype)
    buffer = [weight_update]

    for i, d in enumerate(grad.shape[1:], 1):
        buffer.append(zero_param(ctx, f"dim{i}", one_shape(grad.ndim, d, i), ctx.model.storage_dtype))
        weight_update = jnp.minimum(weight_update, buffer[-1])

    weight_update = weight_update + jnp.square(grad)

    for i in range(grad.ndim):
        new = weight_update.max([j for j in range(grad.ndim) if j != i], keepdims=True)
        ctx.parameters[prefixed_name(ctx, f"dim{i}")] = new

    return grad * stable_rsqrt(weight_update, ctx.optimizer.epsilon)


def small_parameter(param_name: str, grad: jnp.ndarray) -> bool:
    return "norm" in param_name.lower() or "rezero" in param_name.lower() or grad.ndim < 2


@with_context(count=False)
def ema(ctx: Context, inp: jnp.ndarray, step: jnp.ndarray, beta: float, quantize: typing.Optional[bool] = None,
        init_val: typing.Optional[jnp.ndarray] = None, heavyball: bool = None, nesterov: bool = None) -> jnp.ndarray:
    if quantize is None:
        quantize = not small_parameter(ctx.global_prefix, inp)
    if heavyball is None:
        heavyball = ctx.optimizer.heavyball
    if nesterov is None:
        heavyball = ctx.optimizer.nesterov
    state = get_param(ctx, "momentum_buffer", inp.shape, dtype=jnp.bfloat16 if quantize else inp.dtype,
                      init_val=jnp.zeros_like(inp) if init_val is None else init_val)
    new_state = state.astype(inp.dtype) * beta + inp * (1 if heavyball else (1 - beta))
    assign(ctx, "momentum_buffer", new_state)
    if not heavyball:  # non-heavyball momentum needs to be debiased
        new_state = new_state * (1 - beta ** (step + 1))
    if nesterov:
        return new_state * beta + inp
    return new_state


@with_context(count=False)
def square_ema(ctx: Context, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:  # == rmsprop
    buffer = ema(ctx, jnp.square(grad), step, 1 - ctx.optimizer.adam_beta2)
    return stable_rsqrt(buffer, ctx.optimizer.epsilon)


@with_context(count=False)
def adam(ctx: Context, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
    return ema(ctx, grad, step, 1 - ctx.optimizer.adam_beta1) * square_ema(ctx, grad, step)


@with_context(count=False)
def shampoo(ctx: Context, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:  # skipcq: PYL-W0640
    preconditioner = Preconditioner(grad, ctx.optimizer.block_size)
    new_preconditioners = []
    for i, old_stat in enumerate(preconditioner.statistics_from_grad(grad)):
        eye = jnp.eye(old_stat.shape[0], dtype=ctx.model.storage_dtype)
        new_stat = ema(ctx, old_stat, step, 1 - ctx.optimizer.shampoo_beta2, True, init_val=eye * ctx.optimizer.epsilon,
                       nesterov=False, heavyball=False)
        prev_p = get_param(ctx, f'preconditioner_{i}', old_stat.shape, dtype=grad.dtype, init_val=eye)
        if ctx.is_initializing:
            continue

        def _new_precond():
            return fallback_pth_root(prev_p, step, new_stat, preconditioner.exponent_for_preconditioner(),
                                     ctx.optimizer.epsilon)

        new_p = lax.cond((step % ctx.optimizer.statistics_compute_steps) == 0, _new_precond, lambda: prev_p)
        new_preconditioners.append(new_p)
        assign(ctx, f"preconditioner_{i}", new_p)
    if ctx.is_initializing:
        return grad
    return preconditioner.preconditioned_grad(grad, new_preconditioners)


@with_context(count=False)
def grafted_shampoo(ctx: Context, weight_update: jnp.ndarray, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
    shampoo_update = shampoo(ctx, grad, step)
    return graft(weight_update, shampoo_update)


def is_stacked(ctx: Context, param_name: str, val: jnp.ndarray):
    return val.shape[0] == ctx.dims.depth and "/step:" in param_name


def clip_norm(ctx: Context, param_name: str, val: jnp.ndarray, min_norm: float) -> jnp.ndarray:
    val = lax.square(val)
    if is_stacked(ctx, param_name, val):
        val = val.sum(tuple(range(1, val.ndim))).reshape((-1,) + (1,) * (val.ndim - 1))
    else:
        val = val.sum()
    return jnp.maximum(jnp.sqrt(val), min_norm)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    grd_norm = clip_norm(ctx, param_name, grad, ctx.optimizer.epsilon)
    wgt_norm = clip_norm(ctx, param_name, ctx.parameters[param_name], 1e-3)
    grad_scale = jnp.minimum(wgt_norm / grd_norm * ctx.optimizer.gradient_clip, 1)
    return grad * grad_scale


def graft(magnitude: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.sqrt(jnp.square(magnitude).sum() / jnp.maximum(jnp.square(direction).sum(), 1e-16))
    return scale * direction


def get_current_lr(ctx: Context, step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(step, opt.warmup_end).astype(jnp.float32) / opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(step.astype(jnp.float32) - opt.warmup_end)
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], step: jnp.ndarray):
    outer_ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, step)

    for param_name, grad in grads.items():
        if "optimizer" in param_name:
            continue
        ctx = outer_ctx.add_to_prefix(param_name, count=False)
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)
        grad = grad.astype(jnp.float64)

        grad = adaptive_gradient_clipping(ctx, param_name, grad)

        if small_parameter(param_name, grad) or ctx.optimizer.graft_to_adam:  # Do adam update for small parameters
            weight_update = adam(ctx, grad, step)
        else:
            weight_update = sm3(ctx, grad)
        if not small_parameter(param_name, grad):
            if ctx.optimizer.use_shampoo:
                if is_stacked(ctx, param_name, grad):
                    weight_update = jnp.stack([grafted_shampoo(ctx, weight_update, grad[i], step)
                                               for i in range(grad.shape[0])], 0)
                else:
                    weight_update = grafted_shampoo(ctx, weight_update, grad, step)
            weight_update = ema(ctx, weight_update, step, 1 - ctx.optimizer.momentum_beta)
            ctx.parameters[param_name] = (1 + ctx.optimizer.weight_decay * parameter_lr) * ctx.parameters[
                param_name]
        weight_update = weight_update.astype(ctx.parameters[param_name].dtype)
        ctx.parameters[param_name] = weight_update * parameter_lr + ctx.parameters[param_name]
