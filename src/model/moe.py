import jax
from jax import lax, numpy as jnp

from src.backend import with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.conv import conv
from src.model.norm import prenorm, scale_norm_act


def all_to_all(ctx: Context, x: jnp.ndarray, split_axis: int, concat_axis: int) -> jnp.ndarray:
    if ctx.is_initializing:
        return x

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray):
        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            return lax.all_to_all(dy, ParallelAxes.model, concat_axis, split_axis, tiled=True)

        return lax.all_to_all(inp, ParallelAxes.model, split_axis, concat_axis, tiled=True), _grad

    return _fn(x)


@prenorm
@with_context()
def dense_moe(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    devices = jax.device_count()
    big_params = devices * ctx.dims.inner_bottleneck_features
    batch, sequence, features = inp.shape
    sequence_slice = sequence // devices

    inp = conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, features, ctx.dims.inner_bottleneck_features)

    # [Batch, Sequence, Features]  ->  [Batch, SequenceSlice, Features * Devices]
    # In essence, 1) Collect features from all devices + 2) Drop unused sequence elements
    if not ctx.is_initializing:
        inp = inp.reshape(batch, sequence_slice, devices, ctx.dims.inner_bottleneck_features)
        inp = all_to_all(ctx, inp, 2, 3)
        inp = inp.reshape(batch, sequence_slice, big_params)

    # Devices^2 more parameters than normal bottleneck block but only Devices-times more flops due to sparsity above
    inp = scale_norm_act(ctx, inp, big_params)
    inp = conv(ctx, inp, ctx.dims.inner_bottleneck_kernel, big_params, big_params, tied=True)
    inp = scale_norm_act(ctx, inp, big_params)

    # [Batch, SequenceSlice, Features * Devices]  ->  [Batch, Sequence, Features]  (PixelShuffle across devices)
    if not ctx.is_initializing:
        inp = inp.reshape(batch, sequence_slice, 1, big_params)
        inp = all_to_all(ctx, inp, 3, 2)
        inp = inp.reshape(batch, sequence, ctx.dims.inner_bottleneck_features)

    return conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, ctx.dims.inner_bottleneck_features, features)
