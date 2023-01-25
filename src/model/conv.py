from typing import Tuple

import jax
from jax import numpy as jnp

from src.backend import get_param, square_grad, with_context, conv as lax_conv
from src.context import Context


@with_context
def conv_weight(ctx: Context, conv_kernel: int, in_features: int, out_features: int, tied: bool = False
                ) -> Tuple[jax.Array, jax.Array]:
    fan_in = jnp.arange(conv_kernel, 0, -1, dtype=ctx.model.storage_dtype)
    fan_in = (1 - 1 / (conv_kernel * ctx.model.conv_scale + ctx.model.conv_shift)) ** fan_in
    fan_in = fan_in / fan_in.sum()
    fan_in = fan_in.reshape(1, 1, -1)
    return get_param(ctx, "conv_weight", [out_features, conv_kernel, in_features], column_axes=2, lr_scale=fan_in,
                     tied=tied, return_sq=True)


@with_context()
def conv(ctx: Context, inp: jax.Array, conv_kernel: int, in_features: int, out_features: int, tied: bool = False):
    weight, weight_sq = conv_weight(ctx, conv_kernel, in_features, out_features, tied)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,), dtype=inp.dtype)

    def _conv(x, y):
        return lax_conv(x, y, [(conv_kernel - 1, 0)], 1)

    return square_grad(_conv, inp, weight, weight_sq)
