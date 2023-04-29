from typing import Tuple, Dict, Union

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, square_grad, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.linear import read, pointwise
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act
from src.model.reversible import reversible, REVERSIBLE_CTX


@with_context()
def input_embed(ctx: Context, inp: jax.Array) -> jax.Array:
    param, param_sq = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features,
                                return_sq=True)

    def _fn(src, wgt):
        return jnp.take(wgt, src, 0)

    if ctx.is_initializing:
        return _fn(inp, param)

    return square_grad(_fn, inp, param, param_sq)


@with_context
def loss_fn(ctx: Context, src: REVERSIBLE_CTX, tgt: jax.Array) -> Tuple[REVERSIBLE_CTX, jax.Array]:
    xent = cross_entropy_loss(ctx)
    features = ctx.dims.memory_slots * ctx.dims.features
    wgt, wgt_sq = get_param(ctx, "out_embd", [features, ctx.dims.vocab], std=1, scale=1 / jax.device_count(),
                            return_sq=True)

    if ctx.is_initializing:
        return src, jnp.zeros((2,))

    def _xent(x, *args):
        return xent(scale_norm_act(ctx, x, features, act=False, weight=False), *args)

    @jax.custom_gradient
    def _fn(inp: REVERSIBLE_CTX, tgt: jax.Array, p: jax.Array, p_sq: jax.Array):
        def _grad(rev_ctx: REVERSIBLE_CTX, d_loss: jax.Array):
            dx0, x0, dx1, x1 = rev_ctx
            dx, _, d_p, d_p_sq = jax.vjp(_xent, x0 + x1, tgt, p, p_sq)[1](d_loss[0])
            return (dx0 + dx, x0, dx1 + dx, x1), None, d_p, d_p_sq

        return (inp, _xent(inp[0] + inp[2], tgt, p, p_sq)), _grad

    return _fn(src, tgt, wgt, wgt_sq)


@with_context()
def block(ctx: Context):
    name_cache = ctx.name_cache

    def _fn(src: Tuple[REVERSIBLE_CTX, jax.Array], inp: Tuple[jax.Array, jax.Array]):
        inp, tgt = inp
        src, original_loss = src
        inp = input_embed(ctx, inp)
        src = reversible(ctx, read, src, inp)
        src = reversible(ctx, pointwise, src)
        src, loss = loss_fn(ctx, src, tgt)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        return (src, original_loss + loss), None

    return _fn


@with_context()
def body_ctx(ctx: Context, src: jax.Array, tgt: jax.Array) -> jax.Array:
    if ctx.is_initializing:
        ctx.add_depth = True
        ctx.parameters, *src = block(ctx)(src, (src, jnp.zeros([], dtype=jnp.int32)))
        ctx.add_depth = False
        return src

    state = jnp.zeros((ctx.dims.batch, ctx.dims.memory_slots, ctx.dims.features), src.dtype)
    (_, loss), _ = lax.scan(block(ctx), ((state, jnp.zeros_like(state), state, jnp.zeros_like(state)), jnp.zeros((2,))),
                            (src, tgt), ctx.dims.sequence)
    return loss


def compute(params: Dict[str, jax.Array], inp: jax.Array) -> jax.Array:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return body_ctx(ctx, src, tgt)
