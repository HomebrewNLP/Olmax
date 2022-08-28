import typing

import jax
from jax import numpy as jnp

from src.backend import get_param, with_context
from src.context import Context
from src.model.conv import bottleneck_block, pointwise_block
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act
from src.model.qrnn import qrnn_block
from src.model.reversible import reversible, revnet_out


@with_context()
def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    param = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features,
                      lr_scale=ctx.optimizer.input_scale)

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(wgt, src, 0)

    return jax.checkpoint(_fn)(inp, param)


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = (src, zero, src, zero)

    src = [ctx.parameters] + list(src)
    for _ in range(ctx.model.unroll_depth):
        for depth in range(ctx.model.qrnn_frequency):
            src = reversible(ctx, pointwise_block, src)
            src = reversible(ctx, bottleneck_block, src)
            src = reversible(ctx, pointwise_block, src)
            if depth % ctx.model.qrnn_frequency == (ctx.model.qrnn_frequency // 2 - 1):
                src = reversible(ctx, qrnn_block, src)
                # lax.cond could work but requires work on the parameter store
    if ctx.is_initializing:
        ctx.parameters = src[0]

    out = revnet_out(src[1:])
    out = scale_norm_act(ctx, out, ctx.dims.features, act=False)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1,
                    lr_scale=ctx.optimizer.output_scale, scale=1 / ctx.dims.heads)
    if ctx.is_initializing:
        return out
    return out, wgt


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    out = body_ctx(ctx, src)
    if ctx.is_initializing:
        return out
    return cross_entropy_loss(ctx, out, tgt)
