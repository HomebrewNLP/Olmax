import jax
import pytest
from jax import numpy as jnp

from src.context import Context
from src.model.norm import norm_forward, scale_norm_act
from unittests.grad.backend import grad_fn, randn_fn, sample_sizes, trials


def general_test(act: bool, psum: bool, samples: int, dim: int):  # skipcq: PYL-W0640
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for trial in range(trials):
        src = randn(int(samples ** 0.5), int(samples ** 0.5), ctx.dims.features)
        multiplier = jax.device_count() if psum else 1
        out_shape = list(src.shape)[1:]
        out_shape[dim] *= multiplier
        wgt = randn(out_shape[dim])
        wgt_sq = randn(out_shape[dim])
        dy = randn(*out_shape)
        print(dy.shape, src.shape, wgt.shape)
        grad = grad_fn(dy, src, wgt, wgt_sq)

        print(trial)
        shape = (1,) * dim + (-1,) + (1,) * (src.ndim - 2 - dim)
        out0 = grad(lambda x: norm_forward(ctx, x[0], x[1].reshape(shape), bool(psum), act, dim)[0])
        out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, (x[1], x[2]), bool(psum), act, dim))

        assert jnp.allclose(out0[0], out1[0])
        assert jnp.allclose(out0[1], out1[1])


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("samples", sample_sizes)
def test_act(act: bool, samples: int):
    general_test(act, False, samples, 2)


@pytest.mark.parametrize("psum", [False, True])
@pytest.mark.parametrize("samples", sample_sizes)
def test_psum(psum: bool, samples: int):
    general_test(True, psum, samples, 2)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("samples", sample_sizes)
def test_dim(dim: int, samples: int):
    general_test(True, False, samples, dim)
