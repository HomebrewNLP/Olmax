import typing

import jax
from jax import numpy as jnp

from src.constants import ParallelAxes
from src.context import WhileTrainContext
from src.model.main import body_ctx


def get_wctx(config: typing.Optional[typing.Dict[str, typing.Any]] = None):
    wctx = WhileTrainContext(config)
    ctx = wctx.ctx

    ctx.dims.batch = 2
    ctx.dims.depth = 3
    ctx.dims.spatial_mixing_kernel = 8
    ctx.dims.sequence = ctx.dims.spatial_mixing_kernel * 2
    ctx.dims.features = 16
    ctx.dims.pointwise_features = 32
    ctx.dims.inner_bottleneck_features = 8

    return wctx, ctx


def replicate(x: typing.Any) -> typing.Any:
    return jax.device_put_replicated(x, jax.local_devices())


def pmap(config: typing.Optional[typing.Dict[str, typing.Any]]):
    wctx, ctx = get_wctx()
    src = replicate(jnp.zeros((ctx.dims.batch, ctx.dims.sequence), dtype=jnp.int32))
    name_cache = {}

    def _fn(x, cfg):
        wctx, ctx = get_wctx(cfg)
        ctx.is_initializing = config is None
        _ = body_ctx(ctx, x)
        name_cache.update(ctx.name_cache)
        return wctx.serialize()

    out = jax.pmap(_fn, ParallelAxes.model)(src, config)
    return out, name_cache


def test_body():
    export, name_cache1 = pmap(None)
    _, name_cache2 = pmap(export)
    print(name_cache1)
    print(name_cache2)
    assert all(name_cache1[k] == v for k, v in name_cache2.items())
    assert all(name_cache2[k] == v for k, v in name_cache1.items())
