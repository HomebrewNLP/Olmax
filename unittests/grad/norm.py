import jax
import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.norm import norm_forward, scale_norm_act
from unittests.grad.backend import grad_fn, randn_fn, sample_sizes, trials


def general_test(act: bool, psum: int, zero_mean: bool, samples: int, power: int):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    ctx.model.norm.zero_mean = zero_mean
    ctx.model.norm.power = power
    randn = randn_fn()
    for trial in range(trials):
        src = randn(int(samples ** 0.5), int(samples ** 0.5), ctx.dims.features)
        if psum == 0:
            multiplier = 1
        elif psum == 1:
            multiplier = jax.device_count()
        else:
            multiplier = ctx.model.reduced
        wgt = randn(ctx.dims.features * multiplier)
        dy = randn(int(samples ** 0.5), int(samples ** 0.5), ctx.dims.features * multiplier)
        grad = grad_fn(dy, src, wgt)

        print(trial)

        if psum != 2:
            out0 = grad(lambda x: norm_forward(ctx, x[0], x[1], bool(psum), act, depth=None)[0])
            out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, x[1], bool(psum), act, depth=None))

            assert jnp.allclose(out0[0], out1[0])
            assert jnp.allclose(out0[1], out1[1])
        else:
            for i in range(jax.device_count()):
                depth = jnp.zeros([]) + i
                out0 = grad(lambda x: norm_forward(ctx, x[0], x[1], bool(psum), act, depth=depth)[0])
                out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, x[1], bool(psum), act, depth=depth))
                assert jnp.allclose(out0[0], out1[0])
                assert jnp.allclose(out0[1], out1[1])


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("samples", sample_sizes)
def test_act(act: bool, samples: int):
    general_test(act, 0, False, samples, 2)


@pytest.mark.parametrize("psum", [0, 1, 2])
@pytest.mark.parametrize("samples", sample_sizes)
def test_psum(psum: int, samples: int):
    general_test(True, psum, False, samples, 2)


@pytest.mark.parametrize("zero_mean", [True, False])
@pytest.mark.parametrize("samples", sample_sizes)
def test_zero_mean(zero_mean: bool, samples: int):
    general_test(True, 0, zero_mean, samples, 2)


@pytest.mark.parametrize("power", [1, 2, 3, 4])
@pytest.mark.parametrize("samples", sample_sizes)
def test_power(power: int, samples: int):
    general_test(True, 0, False, samples, power)
