from typing import Tuple, Optional, Union, Callable

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, promote_to, stable_rsqrt, with_context
from src.context import Context
from src.model.activate import activate_forward, activate_grad


def prenorm(fn: Callable[[Context, jax.Array], jax.Array]):
    def _fn(ctx: Context, inp: jax.Array, *args) -> jax.Array:
        ctx = ctx.add_to_prefix("prenorm")
        inp = scale_norm_act(ctx, inp, ctx.dims.features, act=False)
        out = fn(ctx, inp, *args)
        return scale_norm_act(ctx, out, ctx.dims.features, act=False)

    return _fn


def norm_forward(ctx: Context, src: jax.Array, wgt: Optional[jax.Array] = None, act: bool = True, dim: int = 2,
                 double: bool = False):
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    src_fp64 = promote_to(src, run_type)
    own_sum = lax.square(src_fp64).sum(dim, keepdims=True)
    std = stable_rsqrt(own_sum, ctx.model.norm.eps)
    out = src_fp64 * std * wgt
    if act and double:
        out = jnp.concatenate([activate_forward(out), activate_forward(-out)], dim)
    elif act:
        out = activate_forward(out)
    out = out.astype(src.dtype)
    return out, std


@with_context()
def scale_norm_act(ctx: Context, inp: jax.Array, feature_dim: int,
                   weight: Union[bool, None, Tuple[jax.Array, jax.Array]] = None, act: bool = True, dim: int = 2,
                   double: bool = False) -> jax.Array:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    if weight is None:
        weight, weight_sq = get_param(ctx, "scale", [feature_dim * (1 + double)], std=0, mean=1, dtype=run_type,
                                      return_sq=True)
    elif weight is False:
        weight_sq = weight = 1
    else:
        weight, weight_sq = weight

    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(src: jax.Array, wgt: jax.Array, _wgt_dummy: jax.Array):
        if isinstance(wgt, jax.Array):
            wgt = wgt.reshape((1,) * dim + (-1,) + (1,) * (src.ndim - 1 - dim))

        out, std = norm_forward(ctx, src, wgt, act, dim, double)

        def _grad(dy: jax.Array) -> Union[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, None, None]]:
            inner_src = src
            src_fp64 = promote_to(inner_src, run_type)
            norm_out = src_fp64 * std
            dy = promote_to(dy, run_type)
            if act:
                bw_out = norm_out * wgt
                if double:
                    dy = dy * activate_grad(bw_out) - dy * activate_grad(-bw_out)
                else:
                    dy *= activate_grad(bw_out)
            d_normed = dy * wgt

            d_std = (d_normed * src_fp64).sum(dim, keepdims=True)  # broadcast forward -> sum backward
            d_std *= std ** 3  # reciprocal + x^(1/pow) -> 1/std^2 * 1/std^(pow-1) * 1/pow
            d_std *= src_fp64  # x^pow -> pow * x^(pow-1), multiply fused with above
            dx = d_normed * std - d_std
            dx = dx.astype(src.dtype)

            if not isinstance(wgt, jax.Array):
                return dx, None, None

            summed = list(range(src.ndim))
            del summed[dim]
            d_wgt = dy * norm_out
            d_wgt_sq = (lax.square(d_wgt).sum(summed) * ctx.dims.batch).reshape((-1,)).astype(run_type)
            d_wgt = d_wgt.sum(summed).reshape((-1,)).astype(run_type)
            return dx, d_wgt, d_wgt_sq

        return out, _grad

    return _fn(inp, weight, weight_sq)
