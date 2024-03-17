import math

import jax
from jax import numpy as jnp, lax

from src.backend import with_context, pattern_match
from src.context import Context
from src.model.conv import conv
from src.model.norm import prenorm, scale_norm_act_conv


@prenorm
@with_context()
def dense_block(ctx: Context, inp: jax.Array, depth: jax.Array) -> jax.Array:
    # Following [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992), we're
    # not increasing the dimensionality in the middle, as the rank doesn't increase -> no useful features are added.

    original_shape = inp.shape
    original_batch, sequence, features = original_shape
    max_dims = math.ceil(math.log(sequence, ctx.dims.features))

    arange = jnp.arange(features)
    mask = arange.reshape(1, -1, 1, 1) >= arange.reshape(1, 1, 1, -1)

    def _get_mix_fn(current_depth: int):
        def _fn(x: jax.Array):
            outer_seq = max(sequence // ctx.dims.features ** (current_depth % max_dims + 1), 1)
            inner_seq = sequence // outer_seq  # == dilation
            inner = lax.broadcast_in_dim(mask, (outer_seq, features, inner_seq, features), (0, 1, 2, 3))
            inner = inner.reshape(1, -1, features)[:, :sequence]

            out = x.reshape(original_batch, outer_seq, features, inner_seq)
            out = jnp.transpose(out, (0, 1, 3, 2))
            out = out.reshape(original_batch, sequence, features)
            padded = lax.pad(out[:, :-inner_seq], jnp.zeros((), dtype=inp.dtype),
                             ((0, 0, 0), (inner_seq, 0, 0), (0, 0, 0)))
            return out * inner, padded

        return _fn

    masked, padded = pattern_match(_get_mix_fn, max_dims, depth, inp)
    inp_glu = inp * lax.pad(inp[:, :-1], jnp.ones((), dtype=inp.dtype), ((0, 0, 0), (1, 0, 0), (0, 0, 0)))

    inp = sum(conv(ctx, i, 5, ctx.dims.features, 4 * ctx.dims.features) for i in (inp, inp_glu, masked, padded))
    return scale_norm_act_conv(ctx, inp, 5, 4 * ctx.dims.features, ctx.dims.features)
