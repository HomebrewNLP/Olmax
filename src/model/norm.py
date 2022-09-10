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
        inp = scale_norm_act(ctx, inp, ctx.dims.features, act=False, init_mean=None)
        out = fn(ctx, inp, *args)
        return scale_norm_act(ctx, out, ctx.dims.features, act=False)

    return _fn


def norm_forward(ctx: Context, src: jnp.ndarray, wgt: typing.Optional[jnp.ndarray] = None, psum: bool = False,
                 act: bool = True):
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    original_dtype = src.dtype
    src_fp64 = promote_to(src, run_type)
    if psum:
        src_fp64 = lax.psum(src_fp64, axis_name=ParallelAxes.model)
    if ctx.model.normalize_mean:
        src_fp64 -= src_fp64.mean(-1, keepdims=True)
    std = stable_rsqrt(jnp.power(jnp.abs(src_fp64), ctx.model.norm_power).sum(-1, keepdims=True), ctx.model.norm_eps,
                       ctx.model.norm_power)
    norm_out = src_fp64 * std
    out = norm_out * wgt.reshape((1,) * (src.ndim - 1) + (-1,))
    if act:
        out = activate_forward(out)
    out = out.astype(original_dtype)
    src_fp64 = src_fp64.astype(original_dtype) if ctx.model.normalize_mean or psum else src
    return out, src_fp64, std


@with_context()
def scale_norm_act(ctx: Context, inp: jnp.ndarray, feature_dim: int, weight: typing.Optional[jnp.ndarray] = None,
                   psum: bool = False, act: bool = True, init_mean: typing.Optional[float] = 1) -> jnp.ndarray:
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    if weight is None:
        if init_mean is None:
            # init to 0 if checkpoint so, new layers get learned slowly (-> rezero but input)
            # 1 otherwise to make sure model can learn
            init_mean = float(not bool(ctx.training.checkpoint_load_path))
        weight = get_param(ctx, "scale", [feature_dim], std=0, mean=init_mean, dtype=run_type,
                           lr_scale=ctx.optimizer.norm_scale)

    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, wgt: jnp.ndarray):
        original_dtype = src.dtype
        out, new_src, std = norm_forward(ctx, src, wgt, psum, act)

        def _grad(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
            src_fp64 = promote_to(new_src, run_type)
            norm_out = src_fp64 * std
            reshaped_weight = wgt.reshape((1,) * (src.ndim - 1) + (-1,))
            dy = promote_to(dy, run_type)
            if act:
                dy = dy * activate_grad(norm_out * reshaped_weight)
            d_wgt = (dy * norm_out).sum(list(range(src.ndim - 1))).reshape((-1,))
            dy = dy * reshaped_weight

            d_std = (dy * src_fp64).sum(-1, keepdims=True)  # broadcast forward -> sum backward
            d_std *= std ** (ctx.model.norm_power + 1)  # reciprocal + x^(1/pow) -> 1/std^2 * 1/std^(pow-1) * 1/pow
            d_std *= src_fp64 ** (ctx.model.norm_power - 1)  # x^pow -> pow * x^(pow-1), multiply fused with above
            if ctx.model.norm_power % 2 != 0:  # x^1, x^3 need to be made non-negative; x^2, x^4 don't
                d_std *= lax.sign(src_fp64)
            dx = dy * std - d_std
            if ctx.model.normalize_mean:
                dx -= dx.mean(-1, keepdims=True)
            if psum:
                dx = lax.psum(dx, axis_name=ParallelAxes.model)
            return dx.astype(original_dtype), d_wgt

        return out, _grad

    return _fn(inp, weight)
