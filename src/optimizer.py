from typing import Optional, Dict

import jax
from jax import lax, numpy as jnp

from src.backend import assign, default, get_param, is_stacked, with_context, stable_rsqrt
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


def get_current_lr(ctx: Context, step: jax.Array) -> jax.Array:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(step, opt.warmup_end).astype(jnp.float64)
    learning_rate /= opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(step.astype(jnp.float64))
    return learning_rate.astype(ctx.model.storage_dtype)


def normalize(x: jax.Array) -> jax.Array:
    return x * lax.rsqrt(lax.square(x).sum(1))


def svd_fisher(ctx: Context, grad: jax.Array):
    key = jax.random.PRNGKey(ctx.seed)
    vectors = normalize(jax.random.normal(key, (ctx.optimizer.svd_components, grad.shape[0])).astype(jnp.float64))
    u = get_param(ctx, "u", vectors.shape[::-1], dtype=ctx.optimizer.momentum_dtype, tied=True,
                  init_val=jnp.zeros_like(vectors)).astype(jnp.float64)
    v = get_param(ctx, "v", vectors.shape, dtype=ctx.optimizer.momentum_dtype, tied=True,
                  init_val=jnp.zeros_like(vectors)).astype(jnp.float64)

    mid = jnp.eye(ctx.optimizer.svd_components * 2 + 1)
    mid = mid.at[:ctx.optimizer.svd_components, :ctx.optimizer.svd_components].set(jnp.transpose(u, (1, 0)) @ u)
    grad = grad * (1 - ctx.optimizer.fisher_decay)
    x0 = jnp.concatenate([u * ctx.optimizer.fisher_decay, grad], 1)
    x0t = jnp.concatenate([v * ctx.optimizer.fisher_decay, grad], 0)
    grad = grad - ((grad @ x0) @ jnp.linalg.inv(jnp.eye(ctx.optimizer.svd_components + 1) + x0t @ x0)) @ x0t

    for i, v in enumerate(vectors, 1):
        local_mid = mid[:ctx.optimizer.svd_components + i, :ctx.optimizer.svd_components + i]
        b0 = normalize(x0 @ local_mid)
        b1 = normalize(x0t)
        inner = b1 @ b0
        for _ in range(ctx.optimizer.log_matrix_power):
            inner = inner @ inner
        v = b0 @ (inner @ (b1 @ v))  # brackets for speed (V=[N,1], b1=[N,K], inner=[K,K], b0=[K,N)
        u = x0 @ (local_mid @ (x0t @ v))
        x0 = jnp.concatenate([x0, u.reshape(-1, 1)], 1)
        x0t = jnp.concatenate([x0t, v.reshape(-1, 1)], 0)
    assign(ctx, "u", x0[:, -vectors:])
    assign(ctx, "v", x0t[-vectors:, :])
    return grad


def update(ctx: Context, grads: Dict[str, jax.Array], step: jax.Array):
    ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, step)
    keys = [k for k in grads.keys() if "optimizer" not in k and not k.endswith('_sq') and not k.endswith('_sq_stacked')]
    grads = jnp.concatenate([adaptive_gradient_clipping(ctx, k, grads[k].reshape(-1), False) for k in keys], 0)

    ctx.name_cache = {}
    ema_gsq = ema(ctx, lax.square(grads), step, 1 - ctx.optimizer.adam_beta2)
    adam = ema(ctx, grads / stable_rsqrt(ema_gsq, ctx.optimizer.epsilon), step, 1 - ctx.optimizer.adam_beta1)
    prec = svd_fisher(ctx, grads)

    if ctx.is_initializing:
        return

    offset = 0
    for param_name in keys:
        param = ctx.parameters[param_name]
        dtype = ctx.parameters[param_name].dtype
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)
        grad = graft(param_name, adam[offset:offset + param.size], prec[offset:offset + param.size]) * parameter_lr
        if not small_parameter(param_name, grad):
            param *= 1 + ctx.optimizer.weight_decay * parameter_lr
        ctx.parameters[param_name] = (param + grad).astype(dtype)
