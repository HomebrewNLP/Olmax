import typing
import warnings

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, is_model, is_stacked, with_context
from src.context import Context
from src.model.conv import dense_block
from src.model.loss import cross_entropy_loss
from src.model.mixer import mix
from src.model.moe import dense_moe
from src.model.norm import scale_norm_act
from src.model.reversible import FourArrays, reversible, revnet_out


@with_context()
def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    param = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1 / ctx.dims.features)

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray) -> jnp.ndarray:
        return jnp.take(wgt, src, 0)

    return jax.checkpoint(_fn)(inp, param)


@with_context()
def pooled_block(ctx: Context, shared_params: typing.Dict[str, jnp.ndarray]):
    name_cache = ctx.name_cache

    def _fn(carry: FourArrays, inp: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]):
        original_parameters = ctx.parameters
        ctx.parameters, depth = inp
        ctx.parameters.update(shared_params)
        depth = depth.reshape([])
        src = [ctx.parameters] + list(carry)
        src = reversible(ctx, dense_block, src)
        src = reversible(ctx, dense_moe, src)
        src = reversible(ctx, dense_block, src)
        src = reversible(ctx, mix, src, depth)
        name_cache.update(ctx.name_cache)
        if ctx.is_initializing:
            return src
        ctx.parameters = original_parameters
        return src[1:], None

    return _fn


def pool_schedule(ctx: Context):
    cases = ctx.dims.up_down // 2
    schedule = []
    for i in range(-cases, cases):
        i = cases - 1 - (abs(i) - int(i < 0))  # range(-2, 2) == [-2, -1, 0, 1]  ->  [0, 1, 1, 0]
        pool = 2 ** i
        pooled = ctx.dims.sequence // pool

        if jax.default_backend() == "tpu" and pooled < 128:
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"minimum (128). Decreasing pooling, but perhaps increase sequence length?")
            pool = ctx.dims.up_down = ctx.dims.sequence // 128
            pooled = 128
        if pooled < jax.device_count():
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"the number of devices ({jax.device_count()}), which is necessary for MoE's all-to-all."
                          f"Decreasing pooling, but perhaps increase sequence length?")
            pool = ctx.dims.up_down = ctx.dims.sequence // jax.device_count()
            pooled = jax.device_count()
        if pooled < ctx.dims.spatial_mixing_kernel:
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"{ctx.dims.spatial_mixing_kernel=}. Decreasing spatial_mixing_kernel, but perhaps increase "
                          f"sequence length?")
            ctx.dims.spatial_mixing_kernel = pooled
        schedule.append(pool)
    return schedule


def merge(original: FourArrays, src: FourArrays, pool: int):
    @jax.custom_gradient
    def _fn(x: FourArrays, y: FourArrays):
        def _grad(dy: FourArrays):
            return dy, [i[::pool] for i in dy]

        out = list(x)  # [x00 (fwd input), x01 (bwd input), x10 (fwd input), x11 (bwd input)]
        out[0] = out[0].at[:, ::pool].add(y[0].astype(out[0].dtype))
        out[2] = out[2].at[:, ::pool].add(y[2].astype(out[2].dtype))
        return tuple(out), _grad

    return _fn(original, src)


@with_context()
def unet(ctx: Context, shared: typing.Dict[str, jnp.ndarray]):
    def _fn(src: FourArrays, inp: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]):
        ctx.parameters, depth = inp
        original_kernel = ctx.dims.spatial_mixing_kernel
        original_depth = ctx.dims.up_down
        ctx.add_depth = ctx.is_initializing
        for i, pool in enumerate(pool_schedule(ctx)):
            ctx.dims.up_down = pool

            src = tuple(s.reshape(ctx.dims.batch, -1, pool, ctx.dims.features) for s in src)
            original_src = tuple(s[:, :, 1:] for s in src)
            src: FourArrays = tuple(s[:, :, 0] for s in src)
            if ctx.is_initializing:
                ctx.parameters, *src = pooled_block(ctx, shared)(src, (ctx.parameters, jnp.zeros([], dtype=jnp.int32)))
                if i == 0:
                    ctx.name_cache_offsets = ctx.name_cache.copy()
            else:
                params = {p: k for p, k in ctx.parameters.items() if is_stacked(p) and k.shape[0] == ctx.dims.up_down}
                src, _ = lax.scan(pooled_block(ctx, shared), src, (params, jnp.arange(ctx.dims.up_down)),
                                  ctx.dims.up_down)
                if i == 0:
                    ctx.name_cache_offsets = ctx.name_cache.copy()
            src = tuple(jnp.concatenate([s, os], 2).reshape(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
                        for s, os in zip(src, original_src))
            ctx.dims.spatial_mixing_kernel = original_kernel
        ctx.dims.up_down = original_depth
        ctx.add_depth = False
        if ctx.is_initializing:
            return ctx.parameters
        return src, None

    return _fn


@with_context()
def stem(ctx: Context, src: FourArrays) -> FourArrays:
    if ctx.is_initializing:
        ctx.parameters = unet(ctx, {})(src, (ctx.parameters, jnp.zeros([], dtype=jnp.int32)))
        for k, v in ctx.parameters.items():
            if is_stacked(k):
                ctx.parameters[k] = jnp.stack([v] * ctx.dims.depth)
        return src

    params = {k: v for k, v in ctx.parameters.items() if is_model(k)}
    shared = {k: v for k, v in params.items() if not is_stacked(k)}
    params = {k: v for k, v in params.items() if is_stacked(k)}
    schedule_length = sum(pool_schedule(ctx))
    depths = jnp.arange(0, ctx.dims.depth * schedule_length, schedule_length)
    src, _ = lax.scan(unet(ctx, shared), src, (params, depths), ctx.dims.depth)
    return src


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = stem(ctx, (src, zero, src, zero))
    out = revnet_out(src)
    out = scale_norm_act(ctx, out, ctx.dims.features, act=False, weight=False)
    wgt = get_param(ctx, "out_embd", [ctx.dims.features, ctx.dims.vocab], std=1, scale=1 / jax.device_count())
    if ctx.is_initializing:
        return out
    return out, wgt


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    out = body_ctx(ctx, src)
    if ctx.is_initializing:
        return out
    return cross_entropy_loss(ctx, out, tgt)
