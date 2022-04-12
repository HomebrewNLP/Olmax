import copy
import math
import operator
import typing

import jax
from jax import lax, numpy as jnp

from src.backend import get_param, matmul, conv as lax_conv
from src.constants import ParallelAxes
from src.context import Context

REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def activate(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return inp
    return jax.nn.leaky_relu(inp, ctx.model.leaky_relu_slope)


def psum(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return inp
    return lax.psum(inp, ParallelAxes.model)


def promote_to(inp: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(inp, jnp.promote_types(dtype, jnp.result_type(inp)))


def scale_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("normalization")
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    weight = get_param(ctx, "scale", [ctx.dims.one], std=0, dtype=run_type)

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, wgt: jnp.ndarray):
        original_dtype = src.dtype
        src_fp32 = promote_to(src, run_type)
        mean = src_fp32.mean(-1, keepdims=True)
        std = lax.rsqrt(jnp.square(src_fp32).mean(-1, keepdims=True) - jnp.square(mean))

        def _grad(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
            src_fp32 = promote_to(src, run_type)
            out = (src_fp32 - mean) * std
            d_wgt = (dy * out).sum().reshape((1,))
            dy = dy * std * (1 + wgt)
            dy -= (dy * out).mean(-1, keepdims=True) * out
            dy -= dy.mean(-1, keepdims=True)
            return dy.astype(original_dtype), d_wgt

        out = (src_fp32 - mean) * std * (1 + wgt)
        return out.astype(original_dtype), _grad

    return _fn(inp, weight)


def pool_heads(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    out = inp
    for shift in [operator.add, operator.sub]:
        for halo_idx in range(1, 1 + ctx.model.device_halo_size // 2):
            permutation = []
            for device_idx in range(ctx.dims.sizes.heads):
                permutation.append((device_idx, shift(device_idx, halo_idx) % ctx.dims.sizes.heads))
            out = out + lax.ppermute(inp, ParallelAxes.model, permutation)
    return out


def rezero(ctx: Context, inp: jnp.ndarray, scale: float = 1) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("rezero")
    scale = get_param(ctx, "scale", [ctx.dims.one], std=0, lr_scale=ctx.model.rezero_lr_scale * scale)
    return inp * scale


def conv(ctx: Context, inp: jnp.ndarray, depthwise: bool, conv_kernel: str, scale: float, in_features: str,
         out_features: str, use_rezero: bool = False):
    weight = get_param(ctx, "weight", [out_features, in_features, conv_kernel], column_axes=2,
                       scale=1 if use_rezero else scale)
    if use_rezero:
        weight = rezero(ctx, weight, scale)
    if ctx.is_initializing:
        return inp
    return lax_conv(inp, weight, [(weight.shape[-1] - 1, 0)], ctx.dims.sizes[out_features] if depthwise else 1)


def full_conv(ctx: Context, inp: jnp.ndarray, scale: float, in_features: str, out_features: str,
              use_rezero: bool = False) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("full_conv")
    return conv(ctx, inp, False, ctx.dims.full_conv_kernel, scale, in_features, out_features, use_rezero)


def depthwise_conv(ctx: Context, inp: jnp.ndarray, scale: float, out_features: str,
                   use_rezero: bool = False) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("depthwise_conv")
    return conv(ctx, inp, True, ctx.dims.depthwise_conv_kernel, scale, ctx.dims.one, out_features, use_rezero)


def output_conv(ctx: Context, inp: jnp.ndarray, features: typing.Optional[str] = None):
    ctx = ctx.add_to_prefix("output_conv")
    return full_conv(ctx, inp, ctx.dims.sizes.depth ** -0.5, ctx.dims.intermediate if features is None else features,
                     ctx.dims.features_per_head, True)


def group_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("group_feed_forward")
    mid = full_conv(ctx, inp, 1 / ctx.model.activation_std, ctx.dims.features_per_head, ctx.dims.intermediate)
    mid = activate(ctx, mid)
    mid = depthwise_conv(ctx, mid, 1 / ctx.model.activation_std, ctx.dims.intermediate)
    mid = activate(ctx, mid)
    return output_conv(ctx, mid)


def activated_allsum(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(src: jnp.ndarray):
        dtype = src.dtype
        src = promote_to(src, jnp.float32)
        out = activate(ctx, psum(ctx, src))
        out = out.astype(dtype)

        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            return jnp.where(out >= 0, dy, ctx.model.leaky_relu_slope * dy).astype(dtype)

        return out, _grad

    return _fn(inp)


def reduced_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("reduced_feed_forward")
    mid = full_conv(ctx, inp, 1 / ctx.model.activation_std / ctx.dims.sizes.heads, ctx.dims.features_per_head,
                    ctx.dims.intermediate)
    mid = activated_allsum(ctx, mid)
    return output_conv(ctx, mid)


def glu_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("glu_feed_forward")

    def _fn():
        return full_conv(ctx, inp, 1 / ctx.model.activation_std / ctx.dims.sizes.heads, ctx.dims.features_per_head,
                         ctx.dims.features_per_head)

    mid = _fn()
    if ctx.model.glu_mode >= 1:
        mid = mid * _fn()
    if ctx.model.glu_mode >= 3:
        mid = mid + _fn()
    mid = activate(ctx, mid)
    if ctx.model.glu_mode >= 2:
        mid = scale_norm(ctx, mid)
    return output_conv(ctx, mid, ctx.dims.features_per_head)


def qrnn(ctx: Context, forget: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dtype = forget.dtype
    forget = promote_to(forget, jnp.float32)
    x = promote_to(x, jnp.float32)
    forget = jax.nn.hard_sigmoid(forget)
    for i in range(int(math.log2(ctx.dims.sizes.sequence))):
        x += jnp.concatenate([jnp.zeros((x.shape[0], 2 ** i, x.shape[2])), x[:, :-2 ** i] * forget[:, 2 ** i:]], 1)
        forget *= jnp.concatenate([jnp.ones((x.shape[0], 2 ** i, x.shape[2])), forget[:, :-2 ** i]], 1)
    return x.astype(dtype)


def qrnn_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    forget = full_conv(ctx, inp, 1 / ctx.model.activation_std, ctx.dims.features_per_head, ctx.dims.features_per_head)
    mid = full_conv(ctx, inp, 1 / ctx.model.activation_std, ctx.dims.features_per_head, ctx.dims.features_per_head)
    out = qrnn(ctx, forget, mid)
    return output_conv(ctx, out, ctx.dims.features_per_head)


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")
    inp_embd = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features_per_head], std=1e-5)
    out = jnp.take(inp_embd, inp, 0)
    return scale_norm(ctx, out)


def output_embed_shard(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    embd = get_param(ctx, "weight", [ctx.dims.features_per_head, ctx.dims.vocab], std=0,
                     lr_scale=1 / ctx.dims.sizes.heads)
    inp = scale_norm(ctx, inp)
    if ctx.is_initializing:
        return inp
    return psum(ctx, matmul(inp, embd))


def reversible(ctx: Context, fn: typing.Callable[[Context, jnp.ndarray], jnp.ndarray],
               src: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
    if ctx.is_initializing:
        params, x00, x01, x10, x11 = src
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, x10)
        ctx.parameters = new_ctx.parameters
        ctx.parameter_dims = new_ctx.parameter_dims
        ctx.name_cache = new_ctx.name_cache
        ctx.prng_key = new_ctx.prng_key
        return new_ctx.parameters, x10, x11, out, x01

    name_cache = copy.deepcopy(ctx.name_cache)

    def base(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
        ctx.name_cache = copy.deepcopy(name_cache)
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, inp)
        ctx.name_cache = new_ctx.name_cache
        return out

    @jax.custom_gradient
    def _fn(params: typing.Dict[str, jnp.ndarray], x0: jnp.ndarray, back_x0: jnp.ndarray, x1: jnp.ndarray,
            back_x1: jnp.ndarray):
        def _grad(dy: REVERSIBLE_CTX) -> typing.Tuple[
            typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            d_params_old, dy0, y0, dy1, y1 = dy
            x0, grad_fn = jax.vjp(base, params, y0)
            d_params, dx0 = grad_fn(dy1)
            d_params = {k: d_params_old.get(k, 0) + d_params.get(k, 0) for k in d_params.keys()}
            return d_params, dy1, y1 - x0, dx0 + dy0, y0

        out = base(params, x1) + x0
        return (params, x1, x1, out, out), _grad

    return _fn(*src)


def z_loss(ctx: Context, src: jnp.ndarray) -> jnp.ndarray:
    # forward: 0 (-> to not change loss)
    # backward: grad(jnp.square(log_z).mean() * ctx.training.z_loss)
    @jax.custom_gradient
    def _fn(inp: jnp.ndarray):
        def _grad(dy):
            return inp * (dy * (ctx.training.z_loss / inp.size))

        return jnp.zeros((), dtype=inp.dtype), _grad

    return _fn(src)


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    src = promote_to(src, jnp.float32)
    max_logit = lax.stop_gradient(src).max(-1, keepdims=True)
    log_z = lax.log(lax.exp(src - max_logit).sum(-1, keepdims=True)) + max_logit
    loss = log_z - jnp.take_along_axis(src, tgt.reshape(*tgt.shape, 1), -1)
    loss = loss.mean()
    accuracy = (jnp.argmax(src, 2) == tgt).astype(jnp.float32).mean()
    if ctx.training.z_loss:
        loss += z_loss(ctx, log_z)
    return loss, accuracy


def revnet_out(src: typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(x0: jnp.ndarray, x0_back: jnp.ndarray, x1: jnp.ndarray, x1_back: jnp.ndarray):
        def _grad(dy) -> typing.Tuple[jnp.ndarray, jnp.ndarray, None, jnp.ndarray]:
            return dy, x0, dy, x1

        return x0 + x1, _grad

    return _fn(*src)


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = (ctx.parameters, src, zero, src, zero)
    for i in range(ctx.dims.sizes.depth):
        src = reversible(ctx, group_feed_forward, src)
        src = reversible(ctx, glu_feed_forward, src)
        src = reversible(ctx, reduced_feed_forward, src)
        src = reversible(ctx, qrnn_block, src)
    ctx.parameters = src[0]
    return output_embed_shard(ctx, revnet_out(src[1:]))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return cross_entropy_loss(ctx, body_ctx(ctx, src), tgt)
