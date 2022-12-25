from typing import Tuple, Optional, Union, Callable

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, promote_to, stable_rsqrt, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.activate import activate_forward, activate_grad


def prenorm(fn: Callable[[Context, jax.Array], jax.Array]):
    def _fn(ctx: Context, inp: jax.Array, *args) -> jax.Array:
        ctx = ctx.add_to_prefix("prenorm")
        inp = scale_norm_act(ctx, inp, ctx.dims.features, act=False)
        out = fn(ctx, inp, *args)
        return scale_norm_act(ctx, out, ctx.dims.features, act=False)

    return _fn


def all_gather(inp: jax.Array, dim: int) -> jax.Array:
    @jax.custom_gradient
    def _fn(x):
        def _grad(dy):
            return lax.psum_scatter(dy, axis_name=ParallelAxes.model, scatter_dimension=dim, tiled=True)

        return lax.all_gather(x, axis_name=ParallelAxes.model, axis=dim, tiled=True), _grad

    return _fn(inp)


def norm_forward(ctx: Context, src: jax.Array, wgt: Optional[jax.Array] = None, psum: bool = False,
                 act: bool = True, dim: int = 2):
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float64)
    original_dtype = src.dtype
    src_fp64 = promote_to(src, run_type)
    own_sum = lax.square(src_fp64).sum(dim, keepdims=True)
    if psum:
        own_sum = lax.psum(own_sum, ParallelAxes.model)
    std = stable_rsqrt(own_sum, ctx.model.norm.eps)
    out = src_fp64 * std * wgt
    if act:
        out = activate_forward(out)
    out = out.astype(original_dtype)
    if psum:
        out = all_gather(out, dim)
    return out, std


@with_context()
def scale_norm_act(ctx: Context, inp: jax.Array, feature_dim: int,
                   weight: Union[bool, None, Tuple[jax.Array, jax.Array]] = None,
                   psum: bool = False, act: bool = True, dim: int = 2) -> jax.Array:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float64)
    if weight is None:
        weight, weight_sq = get_param(ctx, "scale", [feature_dim], std=0, mean=1, dtype=run_type, return_sq=True)
    elif weight is False:
        weight_sq = weight = 1
    else:
        weight, weight_sq = weight

    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(src: jax.Array, wgt: jax.Array, _wgt_dummy: jax.Array):
        original_dtype = src.dtype
        if isinstance(wgt, jax.Array):
            reshaped_weight = wgt.reshape((1,) * dim + (-1,) + (1,) * (src.ndim - 1 - dim))
        else:
            reshaped_weight = wgt

        out, std = norm_forward(ctx, src, reshaped_weight, psum, act, dim)

        def _grad(dy: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array]:
            inner_src = lax.all_gather(src, ParallelAxes.model, axis=dim) if psum else src
            src_fp64 = promote_to(inner_src, run_type)
            norm_out = src_fp64 * std
            dy = promote_to(dy, run_type)
            if act:
                dy = dy * activate_grad(norm_out * reshaped_weight)
            if isinstance(wgt, jax.Array):
                summed = list(range(src.ndim))
                del summed[dim]
                d_wgt = dy * norm_out
                d_wgt_sq = (lax.square(d_wgt).sum(summed) * ctx.dims.batch).reshape((-1,)).astype(run_type)
                d_wgt = d_wgt.sum(summed).reshape((-1,)).astype(run_type)
            else:
                d_wgt = None
                d_wgt_sq = None
            dy = dy * reshaped_weight

            d_std = (dy * src_fp64).sum(dim, keepdims=True)  # broadcast forward -> sum backward
            d_std *= std ** 3  # reciprocal + x^(1/pow) -> 1/std^2 * 1/std^(pow-1) * 1/pow
            d_std *= src_fp64  # x^pow -> pow * x^(pow-1), multiply fused with above
            dx = dy * std - d_std
            if psum:
                dx = lax.psum_scatter(dx, axis_name=ParallelAxes.model, scatter_dimension=dim, tiled=True)
            return dx.astype(original_dtype), d_wgt, d_wgt_sq

        return out, _grad

    return _fn(inp, weight, weight_sq)
