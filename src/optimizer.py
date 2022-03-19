import typing

import jax
from jax import numpy as jnp

from .backend import zero_param, assign, prefixed_name
from .context import Context


def optimizer_rsqrt(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.reciprocal(jnp.maximum(jnp.sqrt(inp), 1e-5))


def weighted_add(x1, x2, alpha):
    return x1 * alpha + x2 * (1 - alpha)


def debias(x: jnp.ndarray, current_step: jnp.ndarray, beta: float) -> jnp.ndarray:
    return x * (1 - beta ** current_step)


def zero_param_like(ctx: Context, new_name: str, original_name: str) -> jnp.ndarray:
    return zero_param(ctx, new_name, ctx.parameter_dims.get(original_name, []))


def one_shape(ndim: int, dim_name: str, dim_idx: int) -> typing.List[str]:
    base = ["one"] * ndim
    base[dim_idx] = dim_name
    return base


def sm3(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("sm3", count=False)
    dims = ctx.parameter_dims[param_name] if param_name in ctx.parameter_dims else ["one"] * grad.ndim
    weight_update = zero_param(ctx, "dim0", one_shape(grad.ndim, dims[0], 0))
    buffer = [weight_update]

    for i, d in enumerate(dims[1:], 1):
        buffer.append(zero_param(ctx, f"dim{i}", one_shape(grad.ndim, d, i)))
        weight_update = jnp.minimum(weight_update, buffer[-1])

    weight_update = weight_update + jnp.square(grad)

    for i in range(grad.ndim):
        new = weight_update.max([j for j in range(grad.ndim) if j != i], keepdims=True)
        ctx.parameters[prefixed_name(ctx, f"dim{i}")] = new

    return grad * optimizer_rsqrt(weight_update)


def momentum(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix(f"momentum", count=False)
    state = zero_param_like(ctx, "momentum_buffer", param_name)
    state = grad + state * (1 - ctx.optimizer.momentum_beta)  # 1st for momentum
    assign(ctx, "momentum_buffer", state)
    return grad + state * (1 - ctx.optimizer.momentum_beta)  # 2nd for nesterov


def ema(ctx: Context, param_name: str, inp: jnp.ndarray, current_step: jnp.ndarray, beta: float,
        prefix: str) -> jnp.ndarray:
    ctx = ctx.add_to_prefix(f"{prefix}_ema", count=False)
    state = zero_param_like(ctx, "momentum_buffer", param_name)
    new_state = weighted_add(state, inp, beta)
    assign(ctx, "momentum_buffer", new_state)
    return debias(new_state, current_step, beta)


def adam(ctx: Context, param_name: str, grad: jnp.ndarray, current_step: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("adam", count=False)
    exp_avg = ema(ctx, param_name, grad, current_step, 1 - ctx.optimizer.adam_beta1, "avg")
    exp_avg_sq = ema(ctx, param_name, jnp.square(grad), current_step, 1 - ctx.optimizer.adam_beta2, "avg_sq")
    return exp_avg * optimizer_rsqrt(exp_avg_sq)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    grd_norm = jnp.maximum(jnp.sqrt(jnp.square(grad).sum()), 1e-6)
    wgt_norm = jnp.maximum(jnp.sqrt(jnp.square(ctx.parameters[param_name]).sum()), 1e-3)
    do_clip = jnp.greater(grd_norm * jnp.reciprocal(wgt_norm), ctx.optimizer.gradient_clip)
    clipped = wgt_norm * jnp.reciprocal(grd_norm) * ctx.optimizer.gradient_clip * grad
    return clipped * do_clip + grad * (1 - do_clip)


def get_current_lr(ctx: Context, current_step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(current_step, opt.warmup_end).astype(jnp.float32) / opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(current_step.astype(jnp.float32) - opt.warmup_end)
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], current_step: jnp.ndarray):
    ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, current_step)

    for param_name, grad in grads.items():
        inner_ctx = ctx.add_to_prefix(param_name, count=False)
        if "optimizer" in param_name:
            continue
        grad = grad.astype(ctx.model.storage_dtype)
        grad = adaptive_gradient_clipping(inner_ctx, param_name, grad)
        if "norm" in param_name.lower() or "rezero" in param_name.lower() or grad.ndim < 2:
            grad = adam(inner_ctx, param_name, grad, current_step)  # Do adam update for small parameters
        else:
            grad = sm3(inner_ctx, param_name, grad)
            grad = momentum(inner_ctx, param_name, grad)
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)
        grad *= parameter_lr
        ctx.parameters[param_name] = grad + (1 + ctx.optimizer.weight_decay * parameter_lr) * ctx.parameters[param_name]
