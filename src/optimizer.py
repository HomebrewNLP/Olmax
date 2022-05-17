import typing

import jax
from jax import numpy as jnp

from .backend import zero_param, assign, prefixed_name, get_param
from .context import Context
from .shampoo import Preconditioner, matrix_inverse_pth_root, select_preconditioner


def optimizer_rsqrt(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.reciprocal(jnp.maximum(jnp.sqrt(inp), 1e-5))


def one_shape(ndim: int, dim_name: int, dim_idx: int) -> typing.List[int]:
    base = [1] * ndim
    base[dim_idx] = dim_name
    return base


def sm3(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("sm3", count=False)
    dims = ctx.parameters[param_name].shape if param_name in ctx.parameters else ["one"] * grad.ndim
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


def small_parameter(param_name: str, grad: jnp.ndarray) -> bool:
    return "norm" in param_name.lower() or "rezero" in param_name.lower() or grad.ndim < 2


def ema(ctx: Context, inp: jnp.ndarray, step: jnp.ndarray, beta: float, prefix: str,
        quantize: typing.Optional[bool] = None, init_val: typing.Optional[jnp.ndarray] = None) -> jnp.ndarray:
    ctx = ctx.add_to_prefix(f"{prefix}_ema", count=False)
    if quantize is None:
        quantize = not small_parameter(ctx.global_prefix, inp)
    state = get_param(ctx, "momentum_buffer", inp.shape, dtype=jnp.bfloat16 if quantize else ctx.model.storage_dtype,
                      init_val=jnp.zeros_like(inp) if init_val is None else init_val)
    new_state = state.astype(jnp.float32) * beta + inp * (1 - beta)
    assign(ctx, "momentum_buffer", new_state)
    return new_state * (1 - beta ** (step + 1))  # debias


def square_ema(ctx: Context, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:  # == rmsprop
    ctx = ctx.add_to_prefix("square_ema", count=False)
    return optimizer_rsqrt(ema(ctx, jnp.square(grad), step, 1 - ctx.optimizer.adam_beta2, "square_ema"))


def adam(ctx: Context, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("adam", count=False)
    return ema(ctx, grad, step, 1 - ctx.optimizer.adam_beta1, "avg") * square_ema(ctx, grad, step)


def shampoo(ctx: Context, param_name: str, grad: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("shampoo", count=False)

    preconditioner = Preconditioner(ctx.parameters[param_name], ctx.optimizer.block_size)
    new_preconditioners = []
    for i, old_stat in enumerate(preconditioner.statistics_from_grad(grad)):
        new_stat = ema(ctx, old_stat, step, 1 - ctx.optimizer.shampoo_beta2, f"statistics_{i}", True,
                       jnp.eye(old_stat.shape[0], dtype=ctx.model.storage_dtype) * ctx.optimizer.epsilon)
        prev_p = get_param(ctx, f'preconditioner_{i}', old_stat.shape, dtype=ctx.model.storage_dtype,
                           init_val=jnp.eye(old_stat.shape[0], dtype=ctx.model.storage_dtype))
        if ctx.is_initializing:
            continue

        new_p, error = matrix_inverse_pth_root(new_stat, preconditioner.exponent_for_preconditioner(),
                                               ridge_epsilon=ctx.optimizer.epsilon)
        new_p = select_preconditioner(error, new_p, prev_p)
        new_preconditioners.append(new_p)
        assign(ctx, f"preconditioner_{i}", new_p)
    if ctx.is_initializing:
        return grad
    return preconditioner.preconditioned_grad(grad, new_preconditioners)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    grd_norm = jnp.maximum(jnp.sqrt(jnp.square(grad).sum()), 1e-6)
    wgt_norm = jnp.maximum(jnp.sqrt(jnp.square(ctx.parameters[param_name]).sum()), 1e-3)
    grad_scale = jnp.minimum(wgt_norm / grd_norm * ctx.optimizer.gradient_clip, 1)
    return grad * grad_scale


def graft(ctx: Context, magnitude: jnp.ndarray, direction: jnp.ndarray) -> jnp.ndarray:
    return direction / jnp.maximum(jnp.linalg.norm(direction), ctx.optimizer.epsilon) * jnp.linalg.norm(magnitude)


def get_current_lr(ctx: Context, step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(step, opt.warmup_end).astype(jnp.float32) / opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** jax.nn.relu(step.astype(jnp.float32) - opt.warmup_end)
    return learning_rate.astype(ctx.model.storage_dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], step: jnp.ndarray):
    ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, step)

    for param_name, grad in grads.items():
        if "optimizer" in param_name:
            continue
        inner_ctx = ctx.add_to_prefix(param_name, count=False)
        parameter_lr = lr * ctx.parameter_variance.get(param_name, 1)
        grad = grad.astype(ctx.model.storage_dtype)
        grad = adaptive_gradient_clipping(ctx, param_name, grad)

        if small_parameter(param_name, grad):  # Do adam update for small parameters
            grad = adam(inner_ctx, grad, step)
        else:  # Do shampoo-sm3 update for large parameters
            grad = graft(ctx, sm3(inner_ctx, param_name, grad), shampoo(inner_ctx, param_name, grad, step))
            grad = ema(inner_ctx, grad, step, 1 - ctx.optimizer.momentum_beta, "momentum")
            ctx.parameters[param_name] = (1 + ctx.optimizer.weight_decay * parameter_lr) * ctx.parameters[param_name]
        ctx.parameters[param_name] = grad * parameter_lr + ctx.parameters[param_name]
