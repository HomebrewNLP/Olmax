import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.activate import activate, activate_forward
from unittests.grad.backend import grad_fn, randn_fn


@pytest.mark.parametrize("samples", [2 ** 6, 2 ** 12])
def test_grad(samples: int, trials: int = 16):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for _ in range(trials):
        inp = randn(samples)
        grad = grad_fn((samples,), inp)
        out0, = grad(lambda x: activate(x[0]))
        out1, = grad(lambda x: activate_forward(x[0]))
        assert jnp.allclose(out0, out1)
