import math

from jax import numpy as jnp

from src.backend import get_param, pattern_match, with_context
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
    max_dims = math.ceil(math.log(ctx.dims.sequence, ctx.dims.spatial_mixing_kernel))

    def _get_mix_fn(current_depth: int):
        def _fn(x: jnp.ndarray):
            batch = max(ctx.dims.sequence // ctx.dims.spatial_mixing_kernel ** (current_depth % max_dims + 1), 1)
            out = x.reshape(ctx.dims.batch * batch, ctx.dims.spatial_mixing_kernel, -1, ctx.dims.features)
            out = jnp.einsum("bkrf,kg->bgrf", out, jnp.triu(wgt0) if ctx.model.autoregressive else wgt0)
            out = activate(out)
            out = jnp.einsum("bkrf,kg->bgrf", out, jnp.triu(wgt1) if ctx.model.autoregressive else wgt1)
            return out.reshape(original_shape)

        return _fn

    return pattern_match(_get_mix_fn, max_dims, depth, inp)
