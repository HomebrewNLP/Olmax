import typing

import jax
from jax import numpy as jnp

from shampoo import shampoo
from .backend import zero_param, assign, prefixed_name
from .context import Context


def optimizer_rsqrt(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.reciprocal(jnp.maximum(jnp.sqrt(inp), 1e-5))


def zero_param_like(ctx: Context, new_name: str, original_name: str, dtype: jnp.dtype) -> jnp.ndarray:
    return zero_param(ctx, new_name, ctx.parameter_dims.get(original_name, []), dtype).astype(jnp.float32)


def one_shape(ndim: int, dim_name: str, dim_idx: int) -> typing.List[str]:
    base = ["one"] * ndim
    base[dim_idx] = dim_name
    return base


def sm3(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("sm3", count=False)
    dims = ctx.parameter_dims[param_name] if param_name in ctx.parameter_dims else ["one"] * grad.ndim
    weight_update = zero_param(ctx, "dim0", one_shape(grad.ndim, dims[0], 0), ctx.model.storage_dtype)
    buffer = [weight_update]

    for i, d in enumerate(dims[1:], 1):
        buffer.append(zero_param(ctx, f"dim{i}", one_shape(grad.ndim, d, i), ctx.model.storage_dtype))
        weight_update = jnp.minimum(weight_update, buffer[-1])

    weight_update = weight_update + jnp.square(grad)

    for i in range(grad.ndim):
        new = weight_update.max([j for j in range(grad.ndim) if j != i], keepdims=True)
        ctx.parameters[prefixed_name(ctx, f"dim{i}")] = new

    return grad * optimizer_rsqrt(weight_update)


def ema(ctx: Context, param_name: str, inp: jnp.ndarray, current_step: jnp.ndarray, beta: float,
        prefix: str, quantize: bool) -> jnp.ndarray:
    ctx = ctx.add_to_prefix(f"{prefix}_ema", count=False)
    state = zero_param_like(ctx, "momentum_buffer", param_name, jnp.bfloat16 if quantize else ctx.model.storage_dtype)
    new_state = state * beta + inp * (1 - beta)
    assign(ctx, "momentum_buffer", new_state)
    return new_state * (1 - beta ** (current_step + 1))  # debias


def adam(ctx: Context, param_name: str, grad: jnp.ndarray, current_step: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("adam", count=False)
    exp_avg = ema(ctx, param_name, grad, current_step, 1 - ctx.optimizer.adam_beta1, "avg", False)
    exp_avg_sq = ema(ctx, param_name, jnp.square(grad), current_step, 1 - ctx.optimizer.adam_beta2, "avg_sq", False)
    return exp_avg * optimizer_rsqrt(exp_avg_sq)



def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    grd_norm = jnp.maximum(jnp.sqrt(jnp.square(grad).sum()), 1e-6)
    wgt_norm = jnp.maximum(jnp.sqrt(jnp.square(ctx.parameters[param_name]).sum()), 1e-3)
    grad_scale = jnp.minimum(wgt_norm / grd_norm * ctx.optimizer.gradient_clip, 1)
    return grad * grad_scale


def get_current_lr(ctx: Context, current_step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(current_step, opt.warmup_end).astype(jnp.float32) / opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(current_step.astype(jnp.float32) - opt.warmup_end)
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], current_step: jnp.ndarray):
    ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, current_step)
    if ctx.is_initializing:
        ctx.parameters['/shampoo/count'] = jnp.zeros()

    for param_name, grad in grads.items():
        inner_ctx = ctx.add_to_prefix(param_name, count=False)
        if "optimizer" in param_name or "shampoo" in param_name:
            continue
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)
        grad = grad.astype(ctx.model.storage_dtype)
        grad = adaptive_gradient_clipping(ctx, param_name, grad)

        if "norm" in param_name.lower() or "rezero" in param_name.lower() or grad.ndim < 2:
            grad = adam(inner_ctx, param_name, grad, current_step)  # Do adam update for small parameters
        else:
            grad = shampoo(inner_ctx, param_name, grad)  # Do shampoo update for large parameters
            ctx.parameters[param_name] = (1 + ctx.optimizer.weight_decay * parameter_lr) * ctx.parameters[param_name]
        grad *= parameter_lr
        ctx.parameters[param_name] = grad + ctx.parameters[param_name]
