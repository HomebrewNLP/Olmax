import jax
import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.activate import activate, activate_forward


@pytest.mark.parametrize("samples", [2 ** 6, 2 ** 12])
def test_grad(samples: int, trials: int = 16):
    ctx = Context()
    ctx.is_initializing = False
    key = jax.random.PRNGKey(0)
    for i in range(trials):
        key, k0 = jax.random.split(key)
        inp = jax.random.uniform(k0, (samples,))
        out0 = jax.grad(lambda x: activate(ctx, x).mean())(inp)
        out1 = jax.grad(lambda x: activate_forward(x).mean())(inp)
        assert jnp.allclose(out0, out1)
