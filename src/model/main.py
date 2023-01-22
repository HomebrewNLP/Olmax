from typing import Tuple, Dict, Union

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, is_model, is_stacked, square_grad, with_context
from src.context import Context
from src.model.conv import dense_block
from src.model.loss import cross_entropy_loss
from src.model.moe import dense_moe
from src.model.norm import scale_norm_act
from src.model.reversible import FourArrays, reversible, revnet_out


@with_context()
def input_embed(ctx: Context, inp: jax.Array) -> jax.Array:
    param, param_sq = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features,
                                return_sq=True)

    def _fn(src, wgt):
        return jnp.take(wgt, src, 0)

    if ctx.is_initializing:
        return _fn(inp, param)

    return square_grad(_fn, inp, param, param_sq)


@with_context()
def block(ctx: Context, shared_params: Dict[str, jax.Array]):
    name_cache = ctx.name_cache

    def _fn(carry: FourArrays, inp: Tuple[Dict[str, jax.Array], jax.Array]):
        original_parameters = ctx.parameters
        ctx.parameters, depth = inp
        ctx.parameters.update(shared_params)
        depth = depth.reshape([])
        src = [ctx.parameters] + list(carry)
        src = reversible(ctx, dense_block, src)
        src = reversible(ctx, dense_moe, src)
        src = reversible(ctx, dense_block, src)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        ctx.parameters = original_parameters
        return src[1:], None

    return _fn


@with_context()
def stem(ctx: Context, src: FourArrays) -> FourArrays:
    if ctx.is_initializing:
        ctx.add_depth = True
        ctx.parameters, *src = block(ctx, {})(src, (ctx.parameters, jnp.zeros([], dtype=jnp.int32)))
        ctx.add_depth = False
        return src

    params = {k: v for k, v in ctx.parameters.items() if is_model(k)}
    shared = {k: v for k, v in params.items() if not is_stacked(k)}
    params = {k: v for k, v in params.items() if is_stacked(k)}
    src, _ = lax.scan(block(ctx, shared), src, (params, jnp.arange(ctx.dims.depth)), ctx.dims.depth)
    return src


def body_ctx(ctx: Context, src: jax.Array) -> Union[
    Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = stem(ctx, (src, zero, src, zero))
    out = revnet_out(src)
    out = scale_norm_act(ctx, out, ctx.dims.features, act=False, weight=False)
    wgt, wgt_sq = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1, scale=1 / jax.device_count(),
                            return_sq=True)
    if ctx.is_initializing:
        return out
    return out, wgt, wgt_sq


def compute(params: Dict[str, jax.Array], inp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    out = body_ctx(ctx, src)
    if ctx.is_initializing:
        return out
    return cross_entropy_loss(ctx, out, tgt)
