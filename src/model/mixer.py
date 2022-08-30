import math

from jax import numpy as jnp

from src.backend import get_param, matmul, with_context
from src.context import Context
from src.model.norm import prenorm, scale_norm_act


@prenorm
@with_context()
def mix(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    original_shape = inp.shape
    items = math.ceil(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))
    max_dim = math.floor(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))
    inp = inp.reshape(ctx.dims.batch, -1, *[ctx.dims.spatial_mixing_kernel] * max_dim, ctx.dims.features)
    inp = inp.transpose(0, items + 1, *range(1, 1 + items))
    shape = inp.shape
    weight_shape = [ctx.dims.spatial_mixing_kernel] * 2
    for i in range(items):
        mask = jnp.triu(jnp.ones(weight_shape)) if ctx.model.autoregressive else None
        wgt = get_param(ctx, f"mix_{i}", weight_shape, scale=mask)
        inp = inp.transpose(0, 1, 1 + items, *range(2, 1 + items))
        inp = inp.reshape(*shape)
        if not ctx.is_initializing:
            inp = matmul(inp, wgt)
        if i != items - 1:
            inp = scale_norm_act(ctx, inp, ctx.dims.spatial_mixing_kernel)
    return inp.transpose(0, *range(2, 2 + items), 1).reshape(original_shape)
