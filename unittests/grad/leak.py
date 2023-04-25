import typing

import jax
import pytest
import tqdm
from jax import lax, numpy as jnp

from src.constants import ParallelAxes
from src.context import Context
from src.model.main import body_ctx
from src.model.reversible import revnet_out
from unittests.grad.backend import grad_fn, randn_fn, trials


def mean(x: jax.Array):
    return (x / x.size).sum()


def initialize(samples: int):
    ctx = Context()
    ctx.dims.sequence = samples // 2
    ctx.dims.batch = 2
    return ctx, randn_fn()


def randn_zero(ctx: Context, randn, zero_from: int):
    dy = randn(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
    dy = dy[:, :, :zero_from, :]

    def _inner_fn(x: jax.Array):
        zeros = jnp.zeros((ctx.dims.batch, ctx.dims.sequence - zero_from, ctx.dims.features))
        return jnp.concatenate([x, zeros], 1)

    return jax.pmap(_inner_fn)(dy)


@pytest.mark.parametrize("samples", [8, 128])
def test(samples: int):
    ctx, randn = initialize(samples)
    ctx.is_initializing = True
    ctx.dims.features = 8
    ctx.dims.inner_bottleneck_features = 4
    ctx.dims.pointwise_features = 16
    ctx.dims.memory_slots = ctx.dims.sequence // 2
    src = randn(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features).astype(jnp.bfloat16)

    def _fn(x: jax.Array):
        body_ctx(ctx, x, jnp.zeros((x.shape[0],), jnp.int32))
        params = ctx.parameters
        ctx.parameters = {}
        return params

    params = jax.pmap(_fn, ParallelAxes.model)(src)
    ctx.is_initializing = False

    def _inner(inp: typing.Tuple[typing.Dict[str, jax.Array], jax.Array]):
        params, x = inp
        ctx.name_cache = {}
        ctx.parameters = params
        out = body_ctx(ctx, (x, jnp.zeros_like(x), x, jnp.zeros_like(x)))
        ctx.parameters = {}
        return revnet_out(out)

    for _ in range(trials):
        for i in tqdm.tqdm(range(1, ctx.dims.sequence + 1)):
            dy = randn_zero(ctx, randn, i)
            d_src = grad_fn(dy, params, src)(_inner)[1]
            d_src = lax.rev(d_src, (2,)) != 0
            seq_grad = d_src.sum((0, 1, 3)) > 0
            print(seq_grad)
            for j, itm in enumerate(seq_grad, 1):
                assert itm == (j > (ctx.dims.sequence - i))
