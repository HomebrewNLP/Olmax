from typing import List, Callable, Optional

import pytest
from jax import numpy as jnp, lax

from src.backend import dot
from src.constants import ParallelAxes
from src.context import Context
from src.model.linear import all2all
from src.model.norm import norm_forward, scale_norm_act, scale_norm_act_linear
from unittests.grad.backend import grad_fn, randn_fn, sample_sizes, trials


def general_test(act: bool, samples: int, dim: int, double: bool, linear: Optional[List[int]] = None,
                 fn: Optional[List[Callable]] = None):  # skipcq: PYL-W0640
    linear = []
    fn = []
    ctx = Context()
    ctx.is_initializing = False
    randn = randn_fn()
    for trial in range(trials):
        src = randn(int(samples ** 0.5), int(samples ** 0.5), ctx.dims.features)
        out_shape = list(src.shape)[1:]
        wgt = randn(out_shape[dim])
        if double:
            out_shape[dim] *= 2
        dy = randn(*out_shape)
        print(dy.shape, src.shape, wgt.shape)

        print(trial)
        shape = (1,) * dim + (-1,) + (1,) * (src.ndim - 2 - dim)
        if linear:
            lin_wgts = [randn(lin, out_shape[-1]) for lin in linear]
            grad = grad_fn([randn(*out_shape[:-1], lin) for lin in linear], src, wgt, *lin_wgts)

            def _fn(x):
                out = norm_forward(ctx, x[0], x[1].reshape(shape), act, dim, double)[0]
                return [dot(f(out), w, -1, 1) for w, f in zip(x[2:], fn)]

            out0 = grad(_fn)

            def _fn(x):
                out = scale_norm_act_linear(ctx, x[0], ctx.dims.features, linear, x[1], x[2:], fn, act=act)
                if len(linear) == 1:
                    return [out]
                return list(out)

            out1 = grad(_fn)
        else:
            grad = grad_fn(dy, src, wgt)

            out0 = grad(lambda x: norm_forward(ctx, x[0], x[1].reshape(shape), act, dim, double)[0])
            out1 = grad(lambda x: scale_norm_act(ctx, x[0], ctx.dims.features, x[1], act, dim, double))

        assert jnp.allclose(out0[0], out1[0])
        assert jnp.allclose(out0[1], out1[1])


@pytest.mark.parametrize("act", [True, False])
@pytest.mark.parametrize("samples", sample_sizes)
def test_act(act: bool, samples: int):
    general_test(act, samples, 2, False)


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("samples", sample_sizes)
def test_dim(dim: int, samples: int):
    general_test(True, samples, dim, False)


@pytest.mark.parametrize("double", [False, True])
@pytest.mark.parametrize("samples", sample_sizes)
def test_double(double: bool, samples: int):
    general_test(True, samples, 2, double)


@pytest.mark.parametrize("linear", [[16, 128], [16, 128, 256]])
@pytest.mark.parametrize("samples", sample_sizes)
def test_linear(linear: List[int], samples: int):
    general_test(True, samples, 2, False, linear)


@pytest.mark.parametrize("linear,fn", [([16, 128], [lambda x: lax.psum(x, ParallelAxes.model)]),
                                       ([16, 128], [all2all])])
@pytest.mark.parametrize("samples", sample_sizes)
def test_transformed_linear(linear: List[int], fn: List[Callable], samples: int):
    general_test(True, samples, 2, False, linear, fn)
