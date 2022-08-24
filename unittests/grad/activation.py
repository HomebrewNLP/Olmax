import jax
import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.activate import activate, activate_forward
from unittests.grad.backend import randn_fn


@pytest.mark.parametrize("samples", [2 ** 6, 2 ** 12])
def test_grad(samples: int, trials: int = 16):
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for i in range(trials):
        inp = randn(samples)
        out0 = jax.grad(lambda x: activate(ctx, x).mean())(inp)
        out1 = jax.grad(lambda x: activate_forward(x).mean())(inp)
        assert jnp.allclose(out0, out1)
