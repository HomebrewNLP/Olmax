import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.activate import activate, activate_forward
from unittests.grad.backend import grad_fn, randn_fn, trials, sample_sizes


@pytest.mark.parametrize("samples", sample_sizes)
def test_grad(samples: int):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for _ in range(trials):
        inp = randn(samples)
        dy = randn(samples)
        grad = grad_fn(dy, inp)
        out0, = grad(lambda x: activate(x[0]))
        out1, = grad(lambda x: activate_forward(x[0]))
        assert jnp.allclose(out0, out1)
