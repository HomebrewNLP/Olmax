import typing

import jax
from jax import lax, numpy as jnp

from .backend import add_sq, assign, get_param, is_stacked, prefixed_name, stable_rsqrt, with_context, zero_param
from .context import Context


def one_shape(ndim: int, dim_name: int, dim_idx: int) -> typing.List[int]:
    base = [1] * ndim
    base[dim_idx] = dim_name
    return base


@with_context()
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
    is_small = "norm" in param_name.lower() or "rezero" in param_name.lower()
    is_small |= grad.ndim < (2 + is_stacked(param_name))
    return is_small


@with_context()
def ema(ctx: Context, inp: jnp.ndarray, step: jnp.ndarray, beta: float, quantize: typing.Optional[bool] = None,
        init_val: typing.Optional[jnp.ndarray] = None, heavyball: bool = None, nesterov: bool = None,
        debias: bool = True) -> jnp.ndarray:
    if quantize is None:
        quantize = not small_parameter(ctx.global_prefix, inp)
    if heavyball is None:
        heavyball = ctx.optimizer.heavyball
    if nesterov is None:
        heavyball = ctx.optimizer.nesterov
    state = get_param(ctx, "momentum_buffer", inp.shape, dtype=jnp.bfloat16 if quantize else inp.dtype,
                      init_val=jnp.zeros_like(inp) if init_val is None else init_val, tied=True)
    new_state = state.astype(inp.dtype) * beta + inp * (1 if heavyball else (1 - beta))
    assign(ctx, "momentum_buffer", new_state)
    if not heavyball and debias:  # non-heavyball momentum needs to be debiased
        new_state = new_state / (1 - beta ** (step + 1))
    if nesterov:
        return new_state * beta + inp
    return new_state


def tg_adam(ctx: Context, param_name: str, grad: jnp.ndarray, tg_grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
    ema_g = ema(ctx, grad, step, 1 - ctx.optimizer.adam_beta1)
    ema_gsq = ema(ctx, grad ** 2, step, 1 - ctx.optimizer.adam_beta1)
    ema_tgsq = ema(ctx, tg_grad, step, 1 - ctx.optimizer.adam_beta3)

    tg_update = ema_g * stable_rsqrt(ema_tgsq, ctx.optimizer.epsilon)
    adam_update = ema_g * stable_rsqrt(ema_gsq, ctx.optimizer.epsilon)
    return graft(param_name, adam_update, tg_update)


def norm(param_name: str, val: jnp.ndarray, is_squared: bool):
    if not is_squared:
        val = lax.square(val)
    if is_stacked(param_name):
        val = val.sum(tuple(range(1, val.ndim))).reshape((-1,) + (1,) * (val.ndim - 1))
    else:
        val = val.sum()
    return val


def clip_norm(param_name: str, val: jnp.ndarray, min_norm: float, is_squared: bool) -> jnp.ndarray:
    return jnp.maximum(jnp.sqrt(norm(param_name, val, is_squared)), min_norm)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray, is_squared: bool) -> jnp.ndarray:
    grad = grad.astype(jnp.float64)
    grd_norm = clip_norm(param_name, grad, ctx.optimizer.epsilon, is_squared)
    wgt_norm = clip_norm(param_name, ctx.parameters[param_name], 1e-3, is_squared)
    grad_scale = jnp.minimum(wgt_norm / grd_norm * ctx.optimizer.gradient_clip, 1)
    return grad * grad_scale


def graft(param_name: str, magnitude: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    scale = jnp.sqrt(norm(param_name, magnitude) / jnp.maximum(norm(param_name, direction), 1e-16))
    return scale * direction


def get_current_lr(ctx: Context, step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(step, opt.warmup_end).astype(jnp.float32)
    learning_rate /= opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(step.astype(jnp.float32))
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], step: jnp.ndarray):
    outer_ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, step)

    for param_name, grad in grads.items():
        if "optimizer" in param_name or param_name.endswith('_sq') or param_name.endswith('_sq_stacked'):
            continue
        ctx = outer_ctx.add_to_prefix(param_name, count=False)
        ctx.name_cache = {}
        dtype = ctx.parameters[param_name].dtype
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)

        grad = adaptive_gradient_clipping(ctx, param_name, grad, False)
        grad_sq = adaptive_gradient_clipping(ctx, param_name, grads[add_sq(param_name)], True)
        weight_update = tg_adam(ctx, param_name, grad, grad_sq, step)

        if not small_parameter(param_name, grad):
            ctx.parameters[param_name] = (1 + ctx.optimizer.weight_decay * parameter_lr) * ctx.parameters[param_name]

        weight_update = weight_update.astype(ctx.parameters[param_name].dtype)
        ctx.parameters[param_name] = weight_update * parameter_lr + ctx.parameters[param_name]
        ctx.parameters[param_name] = ctx.parameters.astype(dtype)
