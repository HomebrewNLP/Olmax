from typing import Tuple, Optional, Union, Callable, List

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, promote_to, stable_rsqrt, with_context, conv as lax_conv
from src.context import Context
from src.model.activate import activate_forward, activate_grad
from src.model.conv import conv_weight


def prenorm(fn: Callable[[Context, jax.Array], jax.Array]):
    def _fn(ctx: Context, inp: jax.Array, *args) -> jax.Array:
        ctx = ctx.add_to_prefix("prenorm")
        inp = scale_norm_act(ctx, inp, ctx.dims.features, act=False)
        out = fn(ctx, inp, *args)
        return scale_norm_act(ctx, out, ctx.dims.features, act=False)

    return _fn


def norm_forward(ctx: Context, src: jax.Array, wgt: Optional[jax.Array] = None, act: bool = True, dim: int = 2,
                 double: bool = False, std: Optional[jax.Array] = None):
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    src_fp64 = promote_to(src, run_type)
    if std is None:
        own_sum = lax.square(src_fp64).sum(dim, keepdims=True)
        std = stable_rsqrt(own_sum, ctx.model.norm.eps)
    norm_out = src_fp64 * std
    out = multiplied = norm_out * wgt
    if act and double:
        out = jnp.concatenate([activate_forward(out), activate_forward(-out)], dim)
    elif act:
        out = activate_forward(out)
    out = out.astype(src.dtype)
    return out, norm_out, multiplied, src_fp64, std


def norm_backward(ctx: Context, src: jax.Array, wgt: jax.Array, std: jax.Array, dy: jax.Array, act: bool,
                  dim: int, double: bool, weight_shape: List[int], run_type: jnp.dtype,
                  src_fp64: Optional[jax.Array] = None, norm_out: Optional[jax.Array] = None,
                  bw_out: Optional[jax.Array] = None):
    src_fp64 = promote_to(src, run_type) if src_fp64 is None else src_fp64
    norm_out = (src_fp64 * std) if norm_out is None else norm_out
    dy = promote_to(dy, run_type)
    if act:
        bw_out = (norm_out * wgt) if bw_out is None else bw_out
        if double:
            dy0, dy1 = jnp.split(dy, 2, dim)
            dy = dy0 * activate_grad(bw_out) - dy1 * activate_grad(-bw_out)  # skipcq: PYL-E1130
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
    d_wgt_sq = (lax.square(d_wgt).sum(summed) * ctx.dims.batch).reshape(weight_shape).astype(run_type)
    d_wgt = d_wgt.sum(summed).reshape(weight_shape).astype(run_type)
    return dx, d_wgt, d_wgt_sq


@with_context()
def scale_norm_act(ctx: Context, inp: jax.Array, feature_dim: int,
                   weight: Union[bool, None, Tuple[jax.Array, jax.Array]] = None, act: bool = True, dim: int = 2,
                   double: bool = False) -> jax.Array:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
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
        if isinstance(wgt, jax.Array):
            wgt = wgt.reshape((1,) * dim + (-1,) + (1,) * (src.ndim - 1 - dim))

        out, _, _, _, std = norm_forward(ctx, src, wgt, act, dim, double)

        def _grad(dy: jax.Array) -> Union[Tuple[jax.Array, jax.Array, jax.Array], Tuple[jax.Array, None, None]]:
            return norm_backward(ctx, src, wgt, std, dy, act, dim, double, _wgt_dummy.shape, run_type)

        return out, _grad

    return _fn(inp, weight, weight_sq)


@with_context()
def scale_norm_act_conv(ctx: Context, inp: jax.Array, kernel: int, in_features: int, out_features: int,
                        tied: bool = False) -> jax.Array:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    scale, scale_sq = get_param(ctx, "scale", [in_features], std=0, mean=1, dtype=run_type, return_sq=True)
    weight, weight_sq = conv_weight(ctx, kernel, in_features, out_features, tied)

    if ctx.is_initializing:
        return inp

    dim = inp.ndim - 1

    def _conv(x, y):
        return lax_conv(x, y, [(kernel - 1, 0)], 1)

    @jax.custom_gradient
    def _fn(src: jax.Array, scl: jax.Array, _scl_dummy: jax.Array, wgt: jax.Array, _wgt_dummy: jax.Array):
        scl = scl.reshape((1,) * dim + (-1,))
        print(src.shape, scale.shape, scl.shape, in_features, out_features)
        out, _, _, _, std = norm_forward(ctx, src, scl, True, dim, False)

        def _grad(dy: jax.Array) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            out2, norm_out, bw_out, src_fp64, _ = norm_forward(ctx, src, scl, True, dim, False, std)
            _, d_wgt_sq = jax.vjp(_conv, lax.square(out2), wgt)[1](lax.square(dy))
            dy, d_wgt = jax.vjp(_conv, out2, wgt)[1](dy)
            dx, d_scl, d_scl_sq = norm_backward(ctx, src, scl, std, dy, True, dim, False, _wgt_dummy.shape, run_type,
                                                src_fp64, norm_out, bw_out)
            return dx, d_scl, d_scl_sq, d_wgt, d_wgt_sq

        return lax_conv(out, wgt, [(kernel - 1, 0)], 1), _grad

    return _fn(inp, scale, scale_sq, weight, weight_sq)
