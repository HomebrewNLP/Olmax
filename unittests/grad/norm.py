import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.norm import norm_forward, scale_norm_act
from unittests.grad.backend import grad_fn, randn_fn


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("psum", [True, False])
@pytest.mark.parametrize("samples", [2 ** 10])
def test_grad(act: bool, psum: bool, samples: int, trials: int = 2):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for trial in range(trials):
        src = randn(samples, ctx.dims.features)
        wgt = jnp.ones_like(randn(ctx.dims.features))
        grad = grad_fn((samples, ctx.dims.features), src, wgt)

        out0 = grad(lambda x: norm_forward(ctx, x[0], x[1], psum, act)[0])
        out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, x[1], psum, act))

        print(trial)
        assert jnp.allclose(out0[0], out1[0])
        assert jnp.allclose(out0[1], out1[1])
