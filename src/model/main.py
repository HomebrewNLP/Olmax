from typing import Tuple, Dict, Union

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, square_grad, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.linear import dense_block
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act


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
def block(ctx: Context):
    name_cache = ctx.name_cache

    def _fn(src: jax.Array, inp: jax.Array):  # TODO: integrate embedding and loss into loop
        out = dense_block(ctx, src, inp)
        src = jnp.concatenate([src[:, :-1], out], 1)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        return src, None

    return _fn


@with_context()
def stem(ctx: Context, src: jax.Array) -> jax.Array:
    if ctx.is_initializing:
        ctx.add_depth = True
        ctx.parameters, *src = block(ctx)(src, (src, jnp.zeros([], dtype=jnp.int32)))
        ctx.add_depth = False
        return src

    src, _ = lax.scan(block(ctx), jnp.zeros_like(src[:, ctx.dims.spatial_mixing_kernel]), src.transpose(1, 0, 2),
                      ctx.dims.sequence)
    return src


def body_ctx(ctx: Context, src: jax.Array) -> Union[Tuple[jax.Array, jax.Array, jax.Array], jax.Array]:
    src = input_embed(ctx, src)
    out = stem(ctx, src)
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
