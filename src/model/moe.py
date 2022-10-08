import jax
from jax import lax, numpy as jnp

from src.backend import pattern_match, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.conv import conv
from src.model.norm import prenorm, scale_norm_act


def all_to_all(ctx: Context, x: jnp.ndarray, split_axis: int, concat_axis: int, device_shift: int) -> jnp.ndarray:
    if ctx.is_initializing:
        return x

    devices = list(range(jax.device_count()))
    devices = devices[device_shift:] + devices[:device_shift]

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray):
        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            return lax.all_to_all(dy, ParallelAxes.model, concat_axis, split_axis, tiled=True,
                                  axis_index_groups=devices)

        return lax.all_to_all(inp, ParallelAxes.model, split_axis, concat_axis, tiled=True,
                              axis_index_groups=devices), _grad

    return _fn(x)


@prenorm
@with_context()
def dense_moe(ctx: Context, inp: jnp.ndarray, depth: jnp.ndarray) -> jnp.ndarray:
    devices = jax.device_count()
    big_params = devices * ctx.dims.inner_bottleneck_features
    sequence_slice = ctx.dims.sequence // devices

    inp = conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, ctx.dims.features, ctx.dims.inner_bottleneck_features)

    # [Batch, Sequence, Features]  ->  [Batch, SequenceSlice, Features * Devices]
    # In essence, 1) Collect features from all devices + 2) Drop unused sequence elements
    if not ctx.is_initializing:
        inp = inp.reshape(ctx.dims.batch, sequence_slice, devices, ctx.dims.inner_bottleneck_features)
        inp = pattern_match(lambda d: lambda x: all_to_all(ctx, x, 2, 3, d), jax.device_count(), depth, inp)
        inp = inp.reshape(ctx.dims.batch, sequence_slice, big_params)

    # Devices^2 more parameters than normal bottleneck block but only Devices-times more flops due to sparsity above
    inp = scale_norm_act(ctx, inp, big_params)
    inp = conv(ctx, inp, ctx.dims.inner_bottleneck_kernel, big_params, big_params, tied=True)
    inp = scale_norm_act(ctx, inp, big_params)

    # [Batch, SequenceSlice, Features * Devices]  ->  [Batch, Sequence, Features]  (PixelShuffle across devices)
    if not ctx.is_initializing:
        inp = inp.reshape(ctx.dims.batch, sequence_slice, 1, big_params)
        inp = pattern_match(lambda d: lambda x: all_to_all(ctx, x, 3, 2, d), jax.device_count(), depth, inp)
        inp = inp.reshape(ctx.dims.batch, ctx.dims.sequence, ctx.dims.inner_bottleneck_features)

    return conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, ctx.dims.inner_bottleneck_features, ctx.dims.features)
