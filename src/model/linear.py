import jax
from jax import numpy as jnp, lax

from src.backend import conv as matmul, get_param, square_grad, with_context
from src.context import Context
from src.model.norm import scale_norm_act


@with_context()
def linear(ctx: Context, inp: jax.Array, in_features: int, out_features: int):
    weight, weight_sq = get_param(ctx, "conv_weight", [out_features, in_features], column_axes=2, return_sq=True)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,), dtype=inp.dtype)

    def _mm(x, y):
        return matmul(x, y)

    return square_grad(_mm, inp, weight, weight_sq)


@with_context()
def dense_block(ctx: Context, past: jnp.ndarray, new: jnp.ndarray, step: jnp.ndarray) -> jax.Array:
    """
    Ensure that new is already pre-communicated
    """
    start = jnp.clip(step - 1, min=0)
    last_item = lax.dynamic_index_in_dim(past, jnp.clip(step - 1, min=0), 1)
    context = lax.dynamic_slice_in_dim(past.mean(-1), start, ctx.dims.spatial_mixing_kernel, 1)
    local_input = jnp.concatenate([last_item, new, context], 1)
    features = ctx.dims.features * 4 + ctx.dims.spatial_mixing_kernel
    inp = scale_norm_act(ctx, local_input, features)
    inp = linear(ctx, inp, 1, features, ctx.dims.pointwise_features)
    inp = scale_norm_act(ctx, inp, ctx.dims.pointwise_features)
    return linear(ctx, inp, 1, ctx.dims.pointwise_features, ctx.dims.features)
