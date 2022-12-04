import typing

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, promote_to, stable_rsqrt, with_context
from src.constants import ParallelAxes
from src.context import Context
from src.model.activate import activate_forward, activate_grad


def prenorm(fn: typing.Callable[[Context, jnp.ndarray], jnp.ndarray]):
    def _fn(ctx: Context, inp: jnp.ndarray, *args) -> jnp.ndarray:
        ctx = ctx.add_to_prefix("prenorm")
        inp = scale_norm_act(ctx, inp, ctx.dims.features, act=False)
        out = fn(ctx, inp, *args)
        return scale_norm_act(ctx, out, ctx.dims.features, act=False)

    return _fn


def all_gather(inp: jnp.ndarray, dim: int) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(x):
        def _grad(dy):
            return lax.psum_scatter(dy, axis_name=ParallelAxes.model, scatter_dimension=dim, tiled=True)

        return lax.all_gather(x, axis_name=ParallelAxes.model, axis=dim, tiled=True), _grad

    return _fn(inp)


def norm_forward(ctx: Context, src: jnp.ndarray, wgt: typing.Optional[jnp.ndarray] = None, psum: bool = False,
                 act: bool = True, dim: int = 2):
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    original_dtype = src.dtype
    src_fp64 = promote_to(src, run_type)
    if psum:
        src_fp64 = all_gather(src_fp64, dim)
    if ctx.model.norm.zero_mean:
        src_fp64 -= src_fp64.mean(dim, keepdims=True)
    std = stable_rsqrt(jnp.power(jnp.abs(src_fp64), ctx.model.norm.power).sum(dim, keepdims=True), ctx.model.norm.eps,
                       ctx.model.norm.power)
    norm_out = src_fp64 * std
    out = norm_out * wgt
    if act:
        out = activate_forward(out)
    out = out.astype(original_dtype)
    src_fp64 = src_fp64.astype(original_dtype) if ctx.model.norm.zero_mean or psum else src
    return out, src_fp64, std


@with_context()
def scale_norm_act(ctx: Context, inp: jnp.ndarray, feature_dim: int,
                   weight: typing.Union[bool, None, typing.Tuple[jnp.ndarray, jnp.ndarray]] = None,
                   psum: bool = False, act: bool = True, dim: int = 2) -> jnp.ndarray:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    if weight is None:
        weight = get_param(ctx, "scale", [feature_dim], std=0, mean=1, dtype=run_type)
        if not ctx.is_initializing:
            weight_sq = get_param(ctx, "scale_sq", dtype=run_type)
    elif weight is False:
        weight_sq = weight = 1
    else:
        weight, weight_sq = weight

    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, wgt: jnp.ndarray, wgt_dummy: jnp.ndarray):
        original_dtype = src.dtype
        if isinstance(wgt, jnp.ndarray):
            reshaped_weight = wgt.reshape((1,) * dim + (-1,) + (1,) * (src.ndim - 1 - dim))
        else:
            reshaped_weight = wgt

        out, new_src, std = norm_forward(ctx, src, reshaped_weight, psum, act, dim)

        def _grad(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            src_fp64 = promote_to(new_src, run_type)
            norm_out = src_fp64 * std
            dy = promote_to(dy, run_type)
            if act:
                dy = dy * activate_grad(norm_out * reshaped_weight)
            if isinstance(wgt, jnp.ndarray):
                summed = list(range(src.ndim))
                del summed[dim]
                d_wgt = dy * norm_out
                d_wgt_sq = lax.square(d_wgt).sum(summed).reshape((-1,))
                d_wgt = ((dy * norm_out).sum(summed) * ctx.dims.batch).reshape((-1,))
            else:
                d_wgt = None
                d_wgt_sq = None
            dy = dy * reshaped_weight

            d_std = (dy * src_fp64).sum(dim, keepdims=True)  # broadcast forward -> sum backward
            d_std *= std ** (ctx.model.norm.power + 1)  # reciprocal + x^(1/pow) -> 1/std^2 * 1/std^(pow-1) * 1/pow
            d_std *= src_fp64 ** (ctx.model.norm.power - 1)  # x^pow -> pow * x^(pow-1), multiply fused with above
            if ctx.model.norm.power % 2 != 0:  # x^1, x^3 need to be made non-negative; x^2, x^4 don't
                d_std *= lax.sign(src_fp64)
            dx = dy * std - d_std
            if ctx.model.norm.zero_mean:
                dx -= dx.mean(dim, keepdims=True)
            if psum:
                dx = lax.psum_scatter(dx, axis_name=ParallelAxes.model, scatter_dimension=dim, tiled=True)
            return dx.astype(original_dtype), d_wgt.astype(run_type), d_wgt_sq.astype(run_type)

        return out, _grad

    return _fn(inp, weight, weight_sq)
