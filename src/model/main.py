import typing

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, with_context
from src.context import Context
from src.model.conv import bottleneck_block, pointwise_block
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act
from src.model.reversible import FourArrays, reversible, revnet_out


@with_context()
def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    param = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features,
                      lr_scale=ctx.optimizer.input_scale)

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(wgt, src, 0)

    return jax.checkpoint(_fn)(inp, param)


@with_context()
def step(ctx: Context):
    def _fn(carry: FourArrays, x: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]) -> FourArrays:
        params, idx = x
        src = [params] + list(carry)
        src = reversible(ctx, pointwise_block, src)
        src = reversible(ctx, bottleneck_block, src)
        src = reversible(ctx, pointwise_block, src)
        # src = lax.cond(idx % ctx.model.qrnn_frequency == (ctx.model.qrnn_frequency // 2 - 1),
        #                lambda s: reversible(ctx, qrnn_block, s), lambda s: s, src)
        ctx.parameters = None
        return src[1:]

    return _fn


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = (src, zero, src, zero)
    if ctx.is_initializing:
        ctx.add_depth = True
        src = step(ctx)(src, ({}, 0))
        ctx.add_depth = False
    else:
        params = {p: k for p, k in ctx.parameters.items() if 'optimizer' not in p and k.shape[0] == ctx.dims.depth}
        src = lax.scan(step(ctx), src, (params, jnp.arange(ctx.dims.depth)), ctx.dims.depth)
    out = revnet_out(src)
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
