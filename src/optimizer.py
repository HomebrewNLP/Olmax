from typing import Optional, Dict

import jax
from jax import lax, numpy as jnp

from src.backend import add_sq, assign, default, get_param, is_stacked, stable_rsqrt, with_context
from src.constants import MomentumType
from src.context import Context


def small_parameter(param_name: str, grad: jax.Array) -> bool:
    param_name = param_name.lower()
    is_small = any(f'{k}' in param_name for k in ("norm", "rezero"))
    is_small |= grad.ndim < (2 + is_stacked(param_name))
    return is_small


@with_context()
def ema(ctx: Context, inp: jax.Array, step: jax.Array, beta: float,
        momentum_type: Optional[MomentumType] = None) -> jax.Array:
    default(momentum_type, ctx.optimizer.momentum_type)
    state = get_param(ctx, "momentum_buffer", inp.shape, dtype=ctx.optimizer.momentum_dtype, tied=True,
                      init_val=jnp.zeros_like(inp))
    if ctx.is_initializing:
        return state

    if momentum_type != MomentumType.heavyball:
        inp *= 1 - beta
    inp = inp.astype(ctx.optimizer.momentum_dtype)
    new_state = state * beta + inp
    assign(ctx, "momentum_buffer", new_state)

    new_state = new_state.astype(jnp.float64)
    if momentum_type == MomentumType.debiased:
        new_state = new_state / (1 - beta ** (step + 1))

    if momentum_type == MomentumType.nesterov:
        return new_state * beta + inp
    return new_state


def norm(param_name: str, val: jax.Array, is_squared: bool = False):
    if not is_squared:
        val = lax.square(val)
    if not is_stacked(param_name):
        return val.sum()
    return val.sum(tuple(range(1, val.ndim))).reshape((-1,) + (1,) * (val.ndim - 1))


def clip_norm(param_name: str, val: jax.Array, min_norm: float, is_squared: bool = False) -> jax.Array:
    return jnp.maximum(jnp.sqrt(norm(param_name, val, is_squared)), min_norm)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jax.Array, is_squared: bool) -> jax.Array:
    grad = grad.astype(jnp.float64)
    grd_norm = clip_norm(param_name, grad, ctx.optimizer.epsilon, is_squared)
    wgt_norm = clip_norm(param_name, ctx.parameters[param_name].astype(jnp.float64), 1e-3)
    grad_scale = jnp.minimum(wgt_norm / grd_norm * ctx.optimizer.gradient_clip, 1)
    return grad * grad_scale


def graft(param_name: str, magnitude: jax.Array, direction: jax.Array) -> jax.Array:
    return direction * jnp.sqrt(norm(param_name, magnitude) / jnp.maximum(norm(param_name, direction), 1e-16))


def tg_adam(ctx: Context, param_name: str, grad: jax.Array, tg_grad: jax.Array, step: jax.Array) -> jax.Array:
    ema_g = ema(ctx, grad, step, 1 - ctx.optimizer.adam_beta1)
    ema_gsq = ema(ctx, grad ** 2, step, 1 - ctx.optimizer.adam_beta2)
    ema_tgsq = ema(ctx, tg_grad, step, 1 - ctx.optimizer.adam_beta3)

    if ctx.is_initializing:
        return grad

    adam_update = ema_g * stable_rsqrt(ema_gsq, ctx.optimizer.epsilon)
    tg_update = ema_g * stable_rsqrt(ema_tgsq, ctx.optimizer.epsilon)
    return graft(param_name, adam_update, tg_update)


def get_current_lr(ctx: Context, step: jax.Array) -> jax.Array:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(step, opt.warmup_end).astype(jnp.float64)
    learning_rate /= opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(step.astype(jnp.float64))
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: Dict[str, jax.Array], step: jax.Array):
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
        weight_update = tg_adam(ctx, param_name, grad, grad_sq, step) * parameter_lr

        if ctx.is_initializing:
            continue

        param = ctx.parameters[param_name].astype(jnp.float64)
        if not small_parameter(param_name, grad):
            param *= 1 + ctx.optimizer.weight_decay * parameter_lr
        ctx.parameters[param_name] = (param + weight_update).astype(dtype)
