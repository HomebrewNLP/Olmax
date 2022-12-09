import typing

import jax
import pytest
from jax import numpy as jnp

from src.constants import ParallelAxes
from src.context import WhileTrainContext
from src.model.main import body_ctx
from src.main import add_zeros


def get_wctx(config: typing.Optional[typing.Dict[str, typing.Any]] = None):
    wctx = WhileTrainContext(config)
    ctx = wctx.ctx

    ctx.dims.batch = 16
    ctx.dims.spatial_mixing_kernel = 8
    ctx.dims.sequence = 128
    ctx.dims.features = 16
    ctx.dims.pointwise_features = 32
    ctx.dims.inner_bottleneck_features = 8

    return wctx, ctx


def replicate(x: typing.Any) -> typing.Any:
    return jax.device_put_replicated(x, jax.local_devices())


def pmap(config: typing.Optional[typing.Dict[str, typing.Any]]):
    _, ctx = get_wctx()
    src = replicate(jnp.zeros((ctx.dims.batch, ctx.dims.sequence), dtype=jnp.int32))
    name_cache = {}
    parameter_usages = {}

    def _fn(x, cfg):
        wctx, ctx = get_wctx(cfg)
        ctx.fail_on_missing_parameter = False
        ctx.is_initializing = config is None
        add_zeros(ctx.parameters)
        _ = body_ctx(ctx, x)
        for k in list(ctx.parameters.keys()):
            if "optimizer" in k or k.endswith('_sq') or k.endswith('_sq_stacked'):
                del ctx.parameters[k]
        name_cache.update(ctx.name_cache)
        parameter_usages.update(ctx.parameter_usages)
        return wctx.serialize()

    out = jax.pmap(_fn, ParallelAxes.model)(src, config)
    return WhileTrainContext(out), name_cache, parameter_usages


class BaseTest:
    def __init__(self):
        self.export1, self.name_cache1, self.usages1 = pmap(None)
        self.export2, self.name_cache2, self.usages2 = pmap(self.export1.serialize())

    @staticmethod
    def check(dict1: typing.Dict[str, typing.Any], dict2: typing.Dict[str, typing.Any],
              cond: typing.Callable[[str, typing.Any, typing.Dict[str, typing.Any]], bool]):
        wrong_in_1 = [k for k, v in dict1.items() if cond(k, v, dict2)]
        wrong_in_2 = [k for k, v in dict2.items() if cond(k, v, dict1)]
        dict1 = {k: f'{str(v)[:10]}...' if len(str(v)) > 12 else str(v) for k, v in dict1.items()}
        dict2 = {k: f'{str(v)[:10]}...' if len(str(v)) > 12 else str(v) for k, v in dict2.items()}
        print(f"{dict1=}\n{dict2=}")
        print()
        if wrong_in_1 or wrong_in_2:
            raise ValueError(f"{wrong_in_1=}\n{wrong_in_2=}")

    def is_in(self, dict1: typing.Dict[str, typing.Any], dict2: typing.Dict[str, typing.Any]):
        self.check(dict1, dict2, lambda k, v, d: k not in d)

    def same_shape(self, dict1: typing.Dict[str, typing.Any], dict2: typing.Dict[str, typing.Any]):
        self.is_in(dict1, dict2)
        self.check(dict1, dict2, lambda k, v, d: v.shape != d[k].shape)

    def equal(self, dict1: typing.Dict[str, typing.Any], dict2: typing.Dict[str, typing.Any]):
        self.is_in(dict1, dict2)
        self.check(dict1, dict2, lambda k, v, d: v != d[k])

    def __call__(self):
        raise NotImplementedError


class NameCache(BaseTest):
    def __call__(self):
        self.equal(self.name_cache1, self.name_cache2)


class ParameterUsage(BaseTest):
    def __call__(self):
        self.equal(self.usages1, self.usages2)


class ParameterShapes(BaseTest):
    def __call__(self):
        self.same_shape(self.export1.ctx.parameters, self.export2.ctx.parameters)


class ParameterVariance(BaseTest):
    def __call__(self):
        self.same_shape(self.export1.ctx.parameter_variance, self.export2.ctx.parameter_variance)


classes = [NameCache, ParameterUsage, ParameterShapes, ParameterVariance]


@pytest.mark.parametrize("cls", classes)
def test(cls: type):
    cls()()


def main():
    for cls in classes:
        test(cls)


if __name__ == '__main__':
    main()
