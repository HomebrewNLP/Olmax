from jax import numpy as jnp

from src.backend import conv as lax_conv, get_param, with_context
from src.context import Context
from src.model.activate import activate
from src.model.norm import prenorm, scale_norm_act


@with_context()
def conv(ctx: Context, inp: jnp.ndarray, conv_kernel: int, in_features: int, out_features: int):
    fan_in = jnp.arange(conv_kernel, 0, -1, dtype=ctx.model.storage_dtype)
    fan_in = (1 - 1 / (conv_kernel * ctx.model.conv_scale + ctx.model.conv_shift)) ** fan_in
    fan_in = fan_in / fan_in.sum()
    fan_in = fan_in.reshape(1, 1, -1)
    weight = get_param(ctx, "weight", [out_features, conv_kernel, in_features], column_axes=2, lr_scale=fan_in)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (out_features,))
    return lax_conv(inp, weight, [(conv_kernel - 1, 0)], 1)


@prenorm
@with_context()
def bottleneck_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    inp = conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, ctx.dims.features, ctx.dims.inner_bottleneck_features)
    inp = scale_norm_act(ctx, inp, ctx.dims.inner_bottleneck_features, psum=True)
    inp = conv(ctx, inp, ctx.dims.inner_bottleneck_kernel, ctx.dims.inner_bottleneck_features,
               ctx.dims.inner_bottleneck_features)
    inp = scale_norm_act(ctx, inp, ctx.dims.inner_bottleneck_features)
    return conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, ctx.dims.inner_bottleneck_features, ctx.dims.features)


@prenorm
@with_context()
def dense_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    inp = conv(ctx, inp, ctx.dims.pointwise_kernel, ctx.dims.features, ctx.dims.pointwise_features)
    inp = activate(inp)
    return conv(ctx, inp, ctx.dims.pointwise_kernel, ctx.dims.pointwise_features, ctx.dims.features)
