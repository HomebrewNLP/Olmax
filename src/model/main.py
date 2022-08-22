import typing

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, matmul, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.conv import bottleneck_block, pointwise_block
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
    src = (ctx.parameters, src, zero, src, zero)
    for i in range(ctx.dims.depth):
        src = reversible(ctx, pointwise_block, src)
        src = reversible(ctx, bottleneck_block, src)
        src = reversible(ctx, pointwise_block, src)
        # src = reversible(ctx, moe, src)
        if i % ctx.model.qrnn_frequency == (ctx.model.qrnn_frequency // 2 - 1):
            src = reversible(ctx, qrnn_block, src)
    ctx.parameters = src[0]
    out = revnet_out(src[1:])
    out = scale_norm_act(ctx, out, ctx.dims.features, act=False)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=0,
                    lr_scale=ctx.optimizer.output_scale / ctx.dims.heads)
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

    out = matmul(out[0], out[1]).astype(jnp.float32)
    out = lax.psum(out, ParallelAxes.model)
    out = out.reshape(-1, ctx.dims.vocab)

    def mean(x: jnp.ndarray) -> jnp.ndarray:
        return (x.astype(jnp.float32) / tgt.size).sum()

    loss = mean(jax.nn.logsumexp(out, -1)) - mean(jnp.take_along_axis(out, tgt.reshape(-1, 1), -1))
    acc = mean(lax.eq(out.argmax(-1), tgt.reshape(-1)))
    return loss, acc
