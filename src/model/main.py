import copy
from typing import Tuple, Dict, Optional

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, with_context
from src.constants import SparseAccess
from src.context import Context
from src.model.linear import read, write
from src.model.loss import cross_entropy_loss
from src.model.norm import scale_norm_act
from src.model.reversible import reversible

SIX_ARRAYS = Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


@with_context()
def loss_fn(ctx: Context, src: SIX_ARRAYS, tgt: jax.Array) -> Tuple[SIX_ARRAYS, jax.Array]:
    xent = cross_entropy_loss(ctx)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1, scale=1 / jax.device_count())

    if ctx.is_initializing:
        return src, jnp.zeros((2,))

    def _xent(x, *args):
        return xent(scale_norm_act(ctx, x, ctx.dims.features, act=False, weight=False), *args)

    @jax.custom_gradient
    def _fn(x: jax.Array, _dx: jax.Array, tgt: jax.Array, p: jax.Array):
        def _grad(dy: Tuple[Tuple[jax.Array, jax.Array], jax.Array]):
            (dx, x), d_loss = dy
            dx, _, d_p = jax.vjp(_xent, x, tgt, p, has_aux=True)[1](d_loss[0])
            return dx, x, None, d_p

        return ((x, _dx), jnp.stack(_xent(x, tgt, p))), _grad

    (x1, dx1), loss = _fn(src[2], src[3], tgt, wgt)
    return (src[0], src[1], x1, dx1, src[4], src[5]), loss


@with_context()
def block(ctx: Context):
    name_cache = ctx.name_cache

    def _fn(src: Tuple[SIX_ARRAYS, jax.Array], inp: Tuple[jax.Array, jax.Array, jax.Array]):
        inp, tgt, position = inp
        src, original_loss = src
        src = (ctx.parameters, *src)
        src = reversible(ctx, read, SparseAccess.read, src, inp, position)
        src = reversible(ctx, write, SparseAccess.write, src, inp, position)
        src, loss = loss_fn(ctx, src[1:], tgt)
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


@jax.custom_gradient
def reversible_output(x: SIX_ARRAYS, loss: jax.Array) -> jax.Array:
    x0, _, x1, _, x2, _ = x

    def _grad(dy):
        return (jnp.zeros_like(x0), x0, jnp.zeros_like(x1), x1, jnp.zeros_like(x2), x2), dy

    return loss, _grad


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

    name_cache = copy.deepcopy(ctx.name_cache)
    for i in range(ctx.dims.sequence):
        ctx.name_cache = copy.deepcopy(name_cache)
        (carry, loss), _ = block(ctx)(carry, (src[i], tgt[i], jnp.full_like((), i, dtype=jnp.int32)))
    loss = reversible_output(carry, loss)
    return loss[0], loss[1]


def compute(params: Dict[str, jax.Array], inp: jax.Array) -> Optional[Tuple[jax.Array, jax.Array]]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return body_ctx(ctx, src, tgt)
