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
            return src[0]
        ctx.parameters = original_parameters
        return src[1:], None

    return _fn


@with_context()
def stem(ctx: Context, src: FourArrays) -> FourArrays:
    cases = ctx.dims.depth // 2
    original_kernel = ctx.dims.spatial_mixing_kernel
    original_depth = ctx.dims.depth
    ctx.add_depth = ctx.is_initializing
    shared = {}
    for i in range(-cases, cases):
        i = cases - 1 - (abs(i) - int(i < 0))  # range(-2, 2) == [-2, -1, 0, 1]  ->  [0, 1, 1, 0]
        pool = ctx.dims.depth = 2 ** i
        pooled = ctx.dims.sequence // pool

        if pooled < 128:
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"minimum (128). Decreasing pooling, but perhaps increase sequence length?")
            pool = ctx.dims.depth = ctx.dims.sequence // 128
            pooled = 128
        if pooled < jax.device_count():
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"the number of devices ({jax.device_count()}), which is necessary for MoE's all-to-all."
                          f"Decreasing pooling, but perhaps increase sequence length?")
            pool = ctx.dims.depth = ctx.dims.sequence // jax.device_count()
            pooled = jax.device_count()
        if pooled < ctx.dims.spatial_mixing_kernel:
            warnings.warn(f"Pooled sequence length ({pooled}) with {ctx.dims.sequence=}, {pool=} and {i=} is below "
                          f"{ctx.dims.spatial_mixing_kernel=}. Decreasing spatial_mixing_kernel, but perhaps increase "
                          f"sequence length?")
            ctx.dims.spatial_mixing_kernel = pooled

        original_src = src
        src: FourArrays = tuple(c[:, ::pool] for c in src)
        if ctx.is_initializing:
            ctx.parameters = pooled_block(ctx, {})(src, (ctx.parameters, jnp.zeros([], dtype=jnp.int32)))
            if i == 0:
                ctx.name_cache_offsets = ctx.name_cache.copy()
        else:
            own_params = {p: k for p, k in ctx.parameters.items() if is_model(p) and p.startswith(ctx.global_prefix)}
            params = {p: k for p, k in own_params.items() if is_stacked(p)}
            shared = {p: k for p, k in own_params.items() if not is_stacked(p)}
            src, _ = lax.scan(pooled_block(ctx, shared), src, (params, jnp.arange(ctx.dims.depth)), ctx.dims.depth)
            if i == 0:
                ctx.name_cache_offsets = ctx.name_cache.copy()
        src: FourArrays = tuple(oc.at[:, ::pool].add(c) for c, oc in zip(src, original_src))
        ctx.dims.spatial_mixing_kernel = original_kernel
    ctx.dims.depth = original_depth
    ctx.add_depth = False
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
