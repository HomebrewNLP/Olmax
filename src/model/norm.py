from typing import Tuple, Optional, Union, Callable, List

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, promote_to, stable_rsqrt, with_context, dot
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


def norm_backward(src: jax.Array, wgt: jax.Array, std: jax.Array, dy: jax.Array, act: bool, dim: int, double: bool,
                  weight_shape: List[int], run_type: jnp.dtype, src_fp64: Optional[jax.Array] = None,
                  norm_out: Optional[jax.Array] = None, bw_out: Optional[jax.Array] = None):
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
        return dx, None

    summed = list(range(src.ndim))
    del summed[dim]
    d_wgt = dy * norm_out
    d_wgt = d_wgt.sum(summed).reshape(weight_shape).astype(run_type)
    return dx, d_wgt


@with_context()
def scale_norm_act(ctx: Context, inp: jax.Array, feature_dim: int,
                   weight: Union[bool, None, Tuple[jax.Array, jax.Array]] = None, act: bool = True, dim: int = -1,
                   double: bool = False) -> jax.Array:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    if weight is None:
        weight = get_param(ctx, "scale", [feature_dim], std=0, mean=1, dtype=run_type)
    elif weight is False:
        weight = 1
    else:
        weight = weight

    if ctx.is_initializing:
        return inp

    dim = dim % inp.ndim

    @jax.custom_gradient
    def _fn(src: jax.Array, wgt: jax.Array):
        if isinstance(wgt, jax.Array):
            wgt = wgt.reshape((1,) * dim + (-1,) + (1,) * (src.ndim - 1 - dim))

        out, _, _, _, std = norm_forward(ctx, src, wgt, act, dim, double)

        def _grad(dy: jax.Array) -> Union[Tuple[jax.Array, jax.Array], Tuple[jax.Array, None]]:
            shp = weight.shape if isinstance(weight, jax.Array) else ()
            return norm_backward(src, wgt, std, dy, act, dim, double, shp, run_type)

        return out, _grad

    return _fn(inp, weight)


@with_context()
def scale_norm_act_linear(ctx: Context, inp: jax.Array, in_features: int, out_features: Union[int, List[int]],
                          transform_fns: Optional[List[Callable]] = None, act: bool = True
                          ) -> Union[jax.Array, Tuple[jax.Array, ...]]:
    if isinstance(out_features, int):
        out_features = [out_features]
    if transform_fns is None:
        transform_fns = []
    transform_fns.extend([lambda x: x] * (len(out_features) - len(transform_fns)))
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    scale = get_param(ctx, "scale", [in_features], std=0, mean=1, dtype=run_type)
    weights = [get_param(ctx, f"weight{i}", [o, in_features]) for i, o in enumerate(out_features)]

    if ctx.is_initializing:
        if len(out_features) == 1:
            return jnp.zeros(list(inp.shape[:-1]) + list(out_features), dtype=inp.dtype)
        return tuple(jnp.zeros(inp.shape[:-1] + (o,), dtype=inp.dtype) for o in out_features)

    dim = inp.ndim - 1

    def _mm(x, w):
        return dot(x, w, -1, -1)

    @jax.custom_gradient
    def _fn(src: jax.Array, scl: jax.Array, wgt: List[jax.Array]):
        scl = scl.reshape((1,) * dim + (-1,))
        out, _, _, _, std = norm_forward(ctx, src, scl, act, dim, False)

        def _grad(dy: jax.Array) -> Tuple[jax.Array, jax.Array, List[jax.Array]]:
            out2, norm_out, bw_out, src_fp64, _ = norm_forward(ctx, src, scl, True, dim, False, std)
            dy, d_wgt = zip(*[jax.vjp(lambda x, y: _mm(fn(x), y), out2, w)[1](tmp)
                              for fn, w, tmp in zip(transform_fns, wgt, dy)])
            dx, d_scl = norm_backward(src, scl, std, dy, act, dim, False, scale.shape, run_type, src_fp64, norm_out,
                                      bw_out)
            return dx, d_scl, list(d_wgt)

        return [_mm(fn(out), w) for fn, w in zip(transform_fns, wgt)], _grad

    out = _fn(inp, scale, weights)
    if len(out) == 1:
        return out[0]
    return tuple(out)
