import math

from jax import numpy as jnp

from src.backend import get_param, pattern_match, with_context, dot
from src.context import Context
from src.model.norm import prenorm, scale_norm_act

@prenorm
@with_context()
def mix(ctx: Context, inp: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
    weight_shape = [ctx.dims.spatial_mixing_kernel] * 2
    wgt0 = get_param(ctx, "mix_0", weight_shape)
    wgt1 = get_param(ctx, "mix_1", weight_shape)
    scale = get_param(ctx, "scale", [ctx.dims.features], std=0, mean=1,
                      dtype=jnp.promote_types(ctx.model.computation_dtype, jnp.float32))
    if ctx.is_initializing:
        return inp

    original_shape = inp.shape
    batch, sequence, features = original_shape
    max_dims = math.ceil(math.log(sequence, ctx.dims.spatial_mixing_kernel))
    original_batch = inp.shape[0]
    if ctx.model.autoregressive:
        wgt0 = jnp.triu(wgt0)
        wgt1 = jnp.triu(wgt1)

    def _get_mix_fn(current_depth: int):
        def _fn(x: jnp.ndarray):
            batch = max(sequence // ctx.dims.spatial_mixing_kernel ** (current_depth % max_dims + 1), 1)
            out = x.reshape(original_batch * batch, ctx.dims.spatial_mixing_kernel, -1)

            # Shape[Batch, Sequence, Features] * Shape[Sequence, Sequence] -> Shape[Batch, Features, Sequence]
            out = dot(out,  wgt0, left_contract_dims=(1,), right_contract_dims=(0,))

            out = scale_norm_act(ctx, out, ctx.dims.features, weight=scale, add_to_prefix=False, dim=1)

            # Shape[Batch, Features, Sequence] * Shape[Sequence, Sequence] -> Shape[Batch, Features, Sequence]
            out = dot(out, wgt1, left_contract_dims=(2,), right_contract_dims=(0,))
            out = out.transpose(0, 2, 1)
            return out.reshape(original_shape)

        return _fn

    return pattern_match(_get_mix_fn, max_dims, depth, inp)