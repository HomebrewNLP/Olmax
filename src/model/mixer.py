import math

from jax import numpy as jnp

from src.backend import get_param, matmul, with_context
from src.context import Context
from src.model.activate import activate
from src.model.norm import prenorm


@prenorm
@with_context()
def mix(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    original_shape = inp.shape
    weight_shape = [ctx.dims.spatial_mixing_kernel] * 2
    mask = jnp.triu(jnp.ones(weight_shape, dtype=ctx.model.computation_dtype)) if ctx.model.autoregressive else 1
    weights = [get_param(ctx, f"mix_{i}", weight_shape, scale=mask) for i in range(ctx.model.mixer_iterations)]
    if ctx.is_initializing:
        return inp

    inp = inp.reshape(ctx.dims.batch, -1, ctx.dims.spatial_mixing_kernel, ctx.dims.features)
    inp = inp.transpose(0, 3, 1, 2)
    shape = inp.shape
    transposed_shape = list(shape)
    transposed_shape[3], transposed_shape[2] = transposed_shape[2], transposed_shape[3]
    for i, wgt in enumerate(weights):
        wgt = wgt
        if i != 0:
            inp = inp.reshape(*transposed_shape)
            inp = activate(ctx, inp)
            inp = inp.transpose(0, 1, 3, 2)
        inp = matmul(inp, wgt)
    for _ in range(len(weights) - 1):
        inp = inp.transpose(0, 1, 3, 2)
        inp = inp.reshape(*shape)
    return inp.transpose(0, 2, 3, 1).reshape(original_shape)
