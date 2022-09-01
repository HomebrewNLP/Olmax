import math

from jax import numpy as jnp, lax

from src.backend import get_param, with_context
from src.context import Context
from src.model.activate import activate
from src.model.norm import prenorm


@prenorm
@with_context()
def mix(ctx: Context, inp: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
    weight_shape = [ctx.dims.spatial_mixing_kernel] * 2
    wgt0 = get_param(ctx, f"mix_0", weight_shape)
    wgt1 = get_param(ctx, f"mix_1", weight_shape)
    if ctx.is_initializing:
        return inp

    original_shape = inp.shape
    max_dims = math.floor(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))
    batch = lax.max(ctx.dims.sequence // ctx.dims.spatial_mixing_kernel ** (depth % max_dims + 1), 1)

    mask = jnp.logical_not(jnp.tri(ctx.dims.spatial_mixing_kernel, k=-1)) if ctx.model.autoregressive else 1
    out = inp.reshape(ctx.dims.batch * batch, ctx.dims.spatial_mixing_kernel, -1, ctx.dims.features)
    out = jnp.einsum("bkrf,kg,kg->bgrf", out, wgt0, mask)
    out = activate(ctx, out)
    out = jnp.einsum("bkrf,kg,kg->bgrf", out, wgt1, mask)
    return out.reshape(original_shape)
