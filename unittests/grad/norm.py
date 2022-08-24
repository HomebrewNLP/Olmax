import jax
import pytest
from jax import numpy as jnp

from src.constants import ParallelAxes
from src.context import Context
from src.model.norm import norm_forward, scale_norm_act
from unittests.grad.backend import randn_fn


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("psum", [True, False])
@pytest.mark.parametrize("samples", [2 ** 6, 2 ** 12])
def test_grad(act: bool, psum: bool, samples: int, trials: int = 16):
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for _ in range(trials):
        src = randn(samples, ctx.dims.features)
        wgt = randn(ctx.dims.features)

        def grad(fn):
            return jax.pmap(jax.grad(fn), ParallelAxes.model)(src, wgt)

        out0 = grad(lambda x, y: norm_forward(ctx, x, y, psum, act)[0].mean())
        out1 = grad(lambda x, y: scale_norm_act(ctx, x, ctx.dims.features, y, psum, act).mean())
        assert jnp.allclose(out0, out1)
