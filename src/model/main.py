from typing import Tuple, Dict

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, with_context
from src.constants import SparseAccess
from src.context import Context
from src.model.linear import read, write
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act
from src.model.reversible import reversible, REVERSIBLE_CTX


@with_context
def loss_fn(ctx: Context, src: REVERSIBLE_CTX, tgt: jax.Array) -> Tuple[REVERSIBLE_CTX, jax.Array]:
    xent = cross_entropy_loss(ctx)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1, scale=1 / jax.device_count())

    if ctx.is_initializing:
        return src, jnp.zeros((2,))

    def _xent(x, *args):
        return xent(scale_norm_act(ctx, x, ctx.dims.features, act=False, weight=False), *args)

    @jax.custom_gradient
    def _fn(inp: REVERSIBLE_CTX, tgt: jax.Array, p: jax.Array):
        def _grad(rev_ctx: REVERSIBLE_CTX, d_loss: jax.Array):
            dx0, x0, dx1, x1, d_sparse, sparse = rev_ctx
            dx, _, d_p = jax.vjp(_xent, x0 + x1, tgt, p)[1](d_loss[0])
            return (dx0 + dx, x0, dx1 + dx, x1, d_sparse, sparse), None, d_p

        return (inp, _xent(inp[0] + inp[2], tgt, p)), _grad

    return _fn(src, tgt, wgt)


@with_context()
def block(ctx: Context):
    name_cache = ctx.name_cache

    def _fn(src: Tuple[REVERSIBLE_CTX, jax.Array], inp: Tuple[jax.Array, jax.Array]):
        inp, tgt, position = inp
        src, original_loss = src
        src = reversible(ctx, read, SparseAccess.read, src, inp, position)
        src = reversible(ctx, write, SparseAccess.write, src, inp, position)
        src, loss = loss_fn(ctx, src, tgt)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        return (src, original_loss + loss), None

    return _fn


@with_context()
def batch_embedding(ctx: Context, name: str, *shape: int) -> Tuple[jax.Array, jax.Array]:
    param = get_param(ctx, name, shape, std=1 / shape[-1])
    param = lax.broadcast_in_dim(param, (ctx.dims.batch, *shape), tuple(range(1, 1 + len(shape))))
    return param, jnp.zeros_like(param)


@with_context()
def body_ctx(ctx: Context, src: jax.Array, tgt: jax.Array) -> jax.Array:
    if ctx.is_initializing:
        ctx.add_depth = True
        ctx.parameters, *src = block(ctx)(src, (src, jnp.zeros([], dtype=jnp.int32)))
        ctx.add_depth = False
        return src

    dense0 = batch_embedding(ctx, "dense0", ctx.dims.features)
    dense1 = batch_embedding(ctx, "dense1", ctx.dims.features)
    sparse = batch_embedding(ctx, "sparse", ctx.dims.memory_slots, ctx.dims.memory_features)
    carry = ((*dense0, *dense1, *sparse), jnp.zeros((2,)))
    (_, loss), _ = lax.scan(block(ctx), carry, (src, tgt, jnp.arange(ctx.dims.sequence)), ctx.dims.sequence)
    return loss


def compute(params: Dict[str, jax.Array], inp: jax.Array) -> jax.Array:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return body_ctx(ctx, src, tgt)
