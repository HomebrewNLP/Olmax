import jax
import numpy as np
import pytest
from jax import lax, numpy as jnp

from src.backend import is_main, matmul
from src.constants import ParallelAxes
from src.context import Context
from src.model.loss import cross_entropy_loss
from unittests.grad.backend import grad_fn, randn_fn, sample_sizes, trials


def mean(x: jnp.ndarray):
    return (x / x.size).sum()


def naive_loss(x, y, z_loss):
    tmp = lax.psum(matmul(x[0], x[1]), ParallelAxes.model)
    lse = jax.nn.logsumexp(tmp, -1)
    pos = mean(lse)
    neg = mean(jnp.take_along_axis(tmp.reshape(-1, tmp.shape[-1]), y.reshape(-1, 1), -1))
    z_loss = mean(lse ** 2 * z_loss)
    z_loss = z_loss - lax.stop_gradient(z_loss)
    return pos - neg + z_loss


def initialize(z_loss: float, samples: int):
    ctx = Context()
    ctx.training.z_loss = z_loss
    ctx.dims.sequence = int(samples ** 0.5)
    ctx.dims.batch = int(samples ** 0.5)

    tgt = jax.random.randint(jax.random.PRNGKey(0), (ctx.dims.batch, ctx.dims.sequence), 0, ctx.dims.vocab)

    return ctx, tgt, randn_fn()


def statistics(name: str, var: jnp.ndarray):
    _fn = jax.pmap(lambda x: (lax.pmax(x.max(), "i"), lax.pmin(x.min(), "i"), lax.pmean(x.mean(), "i"),
                              lax.pmean(jnp.square(x - x.mean()).mean(), "i")), "i")
    if is_main():
        vmax, vmin, vmean, vstd = [float(a[0]) for a in _fn(var)]
        print(f"{name}: max={vmax}, min={vmin}, mean={vmean}, std={vstd}")


def general_value_test(z_loss: float, samples: int, vocab: int):  # skipcq: PYL-W0640
    ctx, tgt, randn = initialize(z_loss, samples)
    ctx.dims.vocab = vocab

    for _ in range(trials):
        src = randn(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
        wgt = randn(ctx.dims.features, ctx.dims.vocab)

        grad0 = float(jax.pmap(lambda x: cross_entropy_loss(ctx, x, tgt)[0], ParallelAxes.model)((src, wgt, wgt))[0])
        grad1 = float(jax.pmap(lambda x: naive_loss(x, tgt, z_loss), ParallelAxes.model)((src, wgt, wgt))[0])
        assert np.isclose(grad0, grad1)


def general_grad_test(z_loss: float, samples: int, vocab: int):  # skipcq: PYL-W0640
    ctx, tgt, randn = initialize(z_loss, samples)
    ctx.dims.vocab = vocab

    for _ in range(trials):
        src = randn(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
        wgt = randn(ctx.dims.features, ctx.dims.vocab)
        dy = randn(2)
        grad = grad_fn(dy, src, wgt)

        grad0 = grad(lambda x: cross_entropy_loss(ctx, x, tgt)[0])
        grad1 = grad(lambda x: naive_loss(x, tgt, z_loss))

        for g0, g1 in zip(grad0, grad1):
            statistics("Grad0", g0)
            statistics("Grad1", g1)
            statistics("abs(Grad0 - Grad1)", jax.pmap(lambda x, y: jnp.abs(x - y), "i")(g0, g1))
            statistics("abs(Grad0 / Grad1)", jax.pmap(lambda x, y: jnp.abs(x / y), "i")(g0, g1))
            allclose = int(jax.pmap(lambda x, y: lax.psum(jnp.allclose(x, y).astype(jnp.float32), "i"), "i")(g0, g1)[0])
            if is_main():
                print(f'{allclose=}/{jax.device_count()}\n')
            assert allclose == jax.device_count()


@pytest.mark.parametrize("z_loss", [1, 0.01, 0])
@pytest.mark.parametrize("samples", sample_sizes)
def test_z_loss_value(z_loss: float, samples: int):
    general_value_test(z_loss, samples, 65536)


@pytest.mark.parametrize("vocab", [256, 65536])
def test_vocab_value(vocab: int, samples: int):
    general_value_test(0.01, samples, vocab)


@pytest.mark.parametrize("z_loss", [1, 0.01, 0])
@pytest.mark.parametrize("samples", sample_sizes)
def test_z_loss_grad(z_loss: float, samples: int):
    general_grad_test(z_loss, samples, 65536)


@pytest.mark.parametrize("vocab", [256, 65536])
@pytest.mark.parametrize("samples", sample_sizes)
def test_vocab_grad(vocab: int, samples: int):
    general_grad_test(0.01, samples, vocab)
