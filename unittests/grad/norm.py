import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.norm import norm_forward, scale_norm_act
from unittests.grad.backend import grad_fn, randn_fn, trials, sample_sizes


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("psum", [True, False])
@pytest.mark.parametrize("zero_mean", [True, False])
@pytest.mark.parametrize("samples", sample_sizes)
@pytest.mark.parametrize("power", [1, 2, 3, 4])
def test_grad(act: bool, psum: bool, zero_mean: bool, samples: int, power: int):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    ctx.model.norm.zero_mean = zero_mean
    ctx.model.norm.power = power
    randn = randn_fn()
    for trial in range(trials):
        src = randn(samples, ctx.dims.features)
        wgt = randn(ctx.dims.features)
        grad = grad_fn((samples, ctx.dims.features), src, wgt)

        out0 = grad(lambda x: norm_forward(ctx, x[0], x[1], psum, act)[0])
        out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, x[1], psum, act))

        print(trial)
        assert jnp.allclose(out0[0], out1[0])
        assert jnp.allclose(out0[1], out1[1])
