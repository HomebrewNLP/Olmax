import typing

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, is_stacked, with_context
from src.context import Context
from src.model.conv import bottleneck_block, dense_block
from src.model.loss import cross_entropy_loss
from src.model.mixer import mix
from src.model.norm import scale_norm_act
from src.model.reversible import FourArrays, reversible, revnet_out


@with_context()
def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    param = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features,
                      lr_scale=ctx.optimizer.scale.input)

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(wgt, src, 0)

    return jax.checkpoint(_fn)(inp, param)


@with_context()
def step(ctx: Context):
    def _fn(carry: FourArrays, inp: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]):
        original_parameters = ctx.parameters
        ctx.parameters, depth = inp
        depth = depth.reshape([])
        src = [ctx.parameters] + list(carry)
        src = reversible(ctx, dense_block, src)
        src = reversible(ctx, bottleneck_block, src)
        src = reversible(ctx, dense_block, src)
        src = reversible(ctx, mix, src, depth)
        if ctx.is_initializing:
            return src[0]
        ctx.parameters = original_parameters
        return src[1:], None

    return _fn


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = (src, zero, src, zero)
    if ctx.is_initializing:
        ctx.add_depth = True
        ctx.parameters = step(ctx)(src, (ctx.parameters, jnp.zeros([], dtype=jnp.int32)))
        ctx.add_depth = False
    else:
        params = {p: k for p, k in ctx.parameters.items() if is_stacked(ctx, p, k)}
        src, _ = lax.scan(step(ctx), src, (params, jnp.arange(ctx.dims.depth)), ctx.dims.depth)
    out = revnet_out(src)
    out = scale_norm_act(ctx, out, ctx.dims.features, act=False)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1,
                    lr_scale=ctx.optimizer.scale.output, scale=1 / ctx.dims.heads)
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
