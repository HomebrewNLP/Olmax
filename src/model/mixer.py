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
    max_dims = math.ceil(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))
    mask = jnp.triu(jnp.ones(weight_shape, dtype=ctx.model.computation_dtype)) if ctx.model.autoregressive else 1
    weights = [get_param(ctx, f"mix_{i}", weight_shape, std=1, scale=ctx.dims.spatial_mixing_kernel ** -0.5)
               for i in range(max_dims)]

    if ctx.is_initializing:
        return inp

    inp = inp.reshape(ctx.dims.batch, -1, *[ctx.dims.spatial_mixing_kernel] * max_dims, ctx.dims.features)
    original_dims = ''.join(chr(ord('a') + i) for i in range(len(inp.ndim)))
    for i, wgt in enumerate(weights):
        new_dims = original_dims[:i + 2] + "z" + original_dims[i + 3:]
        reduced_dim = original_dims[i + 2]
        if i > 0:
            inp = activate(inp)
        inp = jnp.einsum(f"{original_dims},{reduced_dim}z,{reduced_dim}z->{new_dims}", inp, wgt, mask)
    return inp.reshape(original_shape)
