import copy
from typing import Tuple, Dict, Optional

import jax
from jax import lax, numpy as jnp

from src.backend import with_context, SIX_ARRAYS, get_param
from src.constants import SparseAccess
from src.context import Context
from src.model.linear import read, write
from src.model.loss import loss_fn
from src.model.reversible import reversible, revnet_out


@with_context()
def block(ctx: Context):
    name_cache = ctx.name_cache

    def _fn(src: Tuple[SIX_ARRAYS, jax.Array], inp: Tuple[jax.Array, jax.Array, jax.Array]):
        inp, tgt, position = inp
        src, original_loss = src
        for i in range(ctx.dims.depth):
            src = reversible(ctx, read, SparseAccess.read, src, inp, position)
            src = reversible(ctx, write, SparseAccess.write, src, inp, position)
        src, loss = loss_fn(ctx, src, tgt)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        return (src, original_loss + loss / ctx.dims.sequence), None

    return _fn


@with_context()
def batch_embedding(ctx: Context, name: str, *shape: int) -> Tuple[jax.Array, jax.Array]:
    param = get_param(ctx, name, shape, std=1 / shape[-1])
    param = lax.broadcast_in_dim(param, (ctx.dims.batch, *shape), tuple(range(1, 1 + len(shape))))
    return param, jnp.zeros_like(param)


@with_context()
def body_ctx(ctx: Context, src: jax.Array, tgt: jax.Array) -> Optional[Tuple[jax.Array, jax.Array]]:
    dense0 = batch_embedding(ctx, "dense0", ctx.dims.features)
    dense1 = batch_embedding(ctx, "dense1", ctx.dims.features)
    sparse = batch_embedding(ctx, "sparse", ctx.dims.memory_slots, ctx.dims.memory_features)
    carry = ((*dense0, *dense1, *sparse), jnp.zeros((2,), dtype=jnp.float64))
    src = src.transpose(1, 0)
    tgt = tgt.transpose(1, 0)

    if ctx.is_initializing:
        ctx.parameters, *src = block(ctx)(carry, (src[0], tgt[0], jnp.zeros([], dtype=jnp.int32)))
        return

    (out, loss), _ = lax.scan(block(ctx), carry, (src, tgt, jnp.arange(ctx.dims.sequence)))
    loss = revnet_out(out, loss)
    return loss[0], loss[1]


def compute(params: Dict[str, jax.Array], inp: jax.Array) -> Optional[Tuple[jax.Array, jax.Array]]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return body_ctx(ctx, src, tgt)
