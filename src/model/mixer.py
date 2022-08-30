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
    items = math.ceil(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))
    samples = 2 ** (int(math.log2(ctx.dims.sequence)) // items)
    inp = inp.reshape(ctx.dims.batch, -1, ctx.dims.spatial_mixing_kernel, ctx.dims.features)
    inp = inp.transpose(0, 3, 1, 2)
    shape = [ctx.dims.batch, ctx.dims.features, -1, samples ** (items - 1)]
    weight_shape = [ctx.dims.spatial_mixing_kernel] * 2
    for i in range(items):
        mask = jnp.triu(jnp.ones(weight_shape, dtype=ctx.model.computation_dtype)) if ctx.model.autoregressive else 1
        wgt = get_param(ctx, f"mix_{i}", weight_shape) * mask
        if ctx.is_initializing:
            continue
        if i != 0:
            inp = inp.reshape(*shape)
            inp = inp.transpose(0, 1, 3, 2)
            inp = activate(ctx, inp)
        inp = matmul(inp, wgt)
    return inp.transpose(0, 1 + items, *range(2, 1 + items), 1).reshape(original_shape)
