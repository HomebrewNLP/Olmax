import random

import jax
import pytest
from jax import lax, numpy as jnp

from src.backend import is_main, matmul
from src.constants import ParallelAxes
from src.context import Context
from src.model.loss import cross_entropy_loss


def randn(div: float, seed: int, *shape: int):
    fn = jax.pmap(lambda x: jax.random.normal(jax.random.PRNGKey(x + seed), shape) / div)
    local_devices = jax.local_device_count()
    seeds = jnp.arange(local_devices * jax.process_index(), local_devices * (1 + jax.process_index()))
    return fn(seeds)


def mean(x: jnp.ndarray):
    return (x / x.size).sum()


def naive_loss(x, y, z_loss):
    tmp = lax.psum(matmul(x[0], x[1]), ParallelAxes.model)
    lse = jax.nn.logsumexp(tmp, -1)
    pos = mean(lse)
    neg = mean(jnp.take_along_axis(tmp.reshape(-1, tmp.shape[-1]), y.reshape(-1, 1), -1))
    return pos - neg + mean(lse ** 2 * z_loss)


@pytest.mark.parametrize("z_loss", [1, 0.01, 0])
@pytest.mark.parametrize("samples", [2 ** 10, 2 ** 16])
def test_loss(z_loss: float, samples: int, trials: int = 16):
    ctx = Context()
    ctx.training.z_loss = z_loss
    ctx.dims.sequence = int(samples ** 0.5)
    ctx.dims.batch = int(samples ** 0.5)

    key = jax.random.PRNGKey(0)
    rng = random.Random(0)
    k0, k1, k2 = jax.random.split(key, 3)
    tgt = jax.random.randint(k0, (ctx.dims.batch, ctx.dims.sequence), 0, ctx.dims.vocab)

    _fn = jax.pmap(lambda x: (lax.pmax(x.max(), "i"), lax.pmin(x.min(), "i"), lax.pmean(x.mean(), "i"),
                              lax.pmean(jnp.square(x - x.mean()).mean(), "i")), "i")

    def statistics(name: str, var: jnp.ndarray):
        max, min, mean, std = [float(a[0]) for a in _fn(var)]
        if is_main():
            print(f"{name}: {max=}, {min=}, {mean=}, {std=}")

    for i in range(trials):
        if is_main():
            print(f"\n\nAttempt: {i}\n")
        div = (ctx.dims.features * ctx.dims.heads) ** 0.25
        src = randn(div, rng.randint(0, 2 ** 30), ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
        wgt = randn(div, rng.randint(0, 2 ** 30), ctx.dims.features, ctx.dims.vocab)

        grad0 = jax.pmap(jax.grad(lambda x: cross_entropy_loss(ctx, x, tgt)[0]), ParallelAxes.model)((src, wgt))
        grad1 = jax.pmap(jax.grad(lambda x: naive_loss(x, tgt, z_loss)), ParallelAxes.model)((src, wgt))
        for g0, g1 in zip(grad0, grad1):
            statistics("Grad0", g0)
            statistics("Grad1", g1)
            statistics("abs(Grad0 - Grad1)", jax.pmap(lambda x, y: jnp.abs(x - y), "i")(g0, g1))
            statistics("abs(Grad0 / Grad1)", jax.pmap(lambda x, y: jnp.abs(x / y), "i")(g0, g1))
            allclose = int(jax.pmap(lambda x, y: lax.psum(jnp.allclose(x, y).astype(jnp.float32), "i"), "i")(g0, g1)[0])
            if is_main():
                print(f'{allclose=}/{jax.device_count()}\n')
            assert allclose == jax.device_count()
