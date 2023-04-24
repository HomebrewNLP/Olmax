import jax
from jax import numpy as jnp, lax

from src.backend import conv as matmul, get_param, square_grad, with_context
from src.context import Context
from src.model.norm import scale_norm_act
from src.constants import ParallelAxes


@with_context()
def linear(ctx: Context, inp: jax.Array, in_features: int, out_features: int):
    weight, weight_sq = get_param(ctx, "conv_weight", [out_features, in_features], column_axes=2, return_sq=True)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,), dtype=inp.dtype)

    def _mm(x, y):
        return matmul(x, y)

    return square_grad(_mm, inp, weight, weight_sq)


@jax.custom_gradient
def all2all(inp):
    def _grad(dy):
        return lax.all_to_all(dy, ParallelAxes.model, inp.ndim - 1, inp.ndim - 1, tiled=True)

    return lax.all_to_all(inp, ParallelAxes.model, inp.ndim - 1, inp.ndim - 1, tiled=True), _grad


@with_context()
def dense_block(ctx: Context, past: jnp.ndarray, new: jnp.ndarray) -> jax.Array:
    # compute-heavy ops (MLP Mixer + first half of pointwise MLP)
    inp = scale_norm_act(ctx, past.transpose(0, 2, 1), ctx.dims.spatial_mixing_kernel)
    inp = linear(ctx, inp, 1, ctx.dims.spatial_mixing_kernel, ctx.dims.mixing_features)
    inp = inp.reshape(ctx.dims.batch, ctx.dims.features * ctx.dims.mixing_features)
    inp = linear(ctx, inp, 1, ctx.dims.features * ctx.dims.mixing_features, ctx.dims.pointwise_features)  # 500M

    # bandwidth heavy ops (in parallel)
    last_item = past[:, -1]
    local_input = jnp.concatenate([last_item, new], 1)
    local_input = scale_norm_act(ctx, local_input, ctx.dims.features * 2)
    local_input = jnp.concatenate([all2all(local_input), local_input], 1)

    # second half of compute heavy ops / pointwise mlp
    inp += linear(ctx, local_input, 1, ctx.dims.features * 4, ctx.dims.pointwise_features)  # 500M
    inp = scale_norm_act(ctx, inp, ctx.dims.pointwise_features)
    return linear(ctx, inp, 1, ctx.dims.pointwise_features, ctx.dims.features)  # 67M (increase size?)
