import math

import jax
from jax import numpy as jnp

from src.backend import get_param, square_grad, with_context, conv as lax_conv, pattern_match
from src.context import Context
from src.model.norm import prenorm, scale_norm_act


@with_context()
def conv(ctx: Context, inp: jax.Array, conv_kernel: int, in_features: int, out_features: int, tied: bool = False):
    fan_in = jnp.arange(conv_kernel, 0, -1, dtype=ctx.model.storage_dtype)
    fan_in = (1 - 1 / (conv_kernel * ctx.model.conv_scale + ctx.model.conv_shift)) ** fan_in
    fan_in = fan_in / fan_in.sum()
    fan_in = fan_in.reshape(1, 1, -1)
    weight, weight_sq = get_param(ctx, "conv_weight", [out_features, conv_kernel, in_features], column_axes=2,
                                  lr_scale=fan_in, tied=tied, return_sq=True)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,), dtype=inp.dtype)

    def _conv(x, y):
        return lax_conv(x, y, [(conv_kernel - 1, 0)], 1)

    return square_grad(_conv, inp, weight, weight_sq)


@prenorm
@with_context()
def dense_block(ctx: Context, inp: jax.Array, depth: jax.Array) -> jax.Array:
    # Following [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992), we're
    # not increasing the dimensionality in the middle, as the rank doesn't increase -> no useful features are added.

    original_shape = inp.shape
    original_batch, sequence, features = original_shape
    max_dims = math.ceil(math.log(sequence, ctx.dims.features))

    def _get_mix_fn(current_depth: int):
        outer_sequence = max(sequence // ctx.dims.features ** (current_depth % max_dims + 1), 1)
        inner_sequence = sequence // outer_sequence  # == dilation
        pad_len = (features - 1) * inner_sequence

        def _fn(x: jax.Array):
            pad = jnp.zeros((original_batch, pad_len), x.dtype)
            x = jnp.concatenate([pad, x.reshape(original_batch, -1)[:, :-pad_len]], 1)
            out = x.reshape(original_batch, outer_sequence, features, inner_sequence)
            out = jnp.transpose(out, (0, 1, 3, 2))
            return out.reshape(original_batch, sequence, features)

        return _fn

    inp = jnp.concatenate([inp, pattern_match(_get_mix_fn, max_dims, depth, inp)], -1)

    inp = conv(ctx, inp, 5, 2 * ctx.dims.features, 2 * ctx.dims.features)
    inp = scale_norm_act(ctx, inp, 2 * ctx.dims.features, double=True)
    return conv(ctx, inp, 5, 4 * ctx.dims.features, ctx.dims.features)
