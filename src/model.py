import copy
import math
import typing

import jax
from jax import lax, numpy as jnp
from jax.experimental.compilation_cache import compilation_cache

from src.backend import get_param, matmul, conv as lax_conv
from src.constants import ParallelAxes
from src.context import Context

compilation_cache.initialize_cache("compilation_cache")
REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def activate(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return inp
    return jax.nn.leaky_relu(inp, ctx.model.leaky_relu_slope)


def promote_to(inp: jnp.ndarray, dtype: jnp.dtype) -> jnp.ndarray:
    return jnp.asarray(inp, jnp.promote_types(dtype, jnp.result_type(inp)))


def scale_norm_act(ctx: Context, inp: jnp.ndarray, weight: typing.Optional[jnp.ndarray] = None, psum: bool = False
                   ) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("normalization")
    run_type = jnp.promote_types(ctx.model.computation_dtype, jnp.float32)
    if weight is None:
        weight = get_param(ctx, "scale", [ctx.dims.one], std=0, dtype=run_type)

    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, wgt: jnp.ndarray):
        original_dtype = src.dtype
        src_fp32 = promote_to(src, run_type)
        if psum:
            src_fp32 = lax.psum(src_fp32, axis_name=ParallelAxes.model)
        mean = src_fp32.mean(-1, keepdims=True)
        var = jnp.maximum(jnp.square(src_fp32).mean(-1, keepdims=True) - jnp.square(mean), ctx.model.norm_eps)
        scale = lax.rsqrt(var) * (1 + wgt)
        out = (src_fp32 - mean) * scale
        out = activate(ctx, out)
        out = out.astype(original_dtype)

        def _grad(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
            out_fp32 = promote_to(out, run_type)
            dy = promote_to(dy, run_type)
            mask = out_fp32 > 0
            out_fp32 = jnp.where(mask, out, out / ctx.model.leaky_relu_slope)
            dy = jnp.where(mask, dy, dy * ctx.model.leaky_relu_slope)
            d_wgt = (dy * out_fp32).sum().reshape((1,))
            dy = dy * scale
            dy -= (dy * out_fp32).mean(-1, keepdims=True) * out_fp32
            dy -= dy.mean(-1, keepdims=True)
            return dy.astype(original_dtype), d_wgt

        return out, _grad

    return _fn(inp, weight)


def rezero(ctx: Context, inp: jnp.ndarray, scale: float = 1) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("rezero")
    scale = get_param(ctx, "scale", [ctx.dims.one], std=0, lr_scale=ctx.model.rezero_lr_scale * scale)
    return inp * scale


def conv(ctx: Context, inp: jnp.ndarray, conv_kernel: str, scale: float, in_features: str, out_features: str):
    ctx = ctx.add_to_prefix("conv")
    weight = get_param(ctx, "weight", [out_features, in_features, conv_kernel], column_axes=2, scale=scale)
    if ctx.is_initializing:
        return jnp.zeros(inp.shape[:-1] + (ctx.dims.sizes[out_features],))
    return lax_conv(inp, weight, [(weight.shape[-1] - 1, 0)], 1)


def bottleneck_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("bottleneck")
    inp = scale_norm_act(ctx, inp)
    inp = conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, 1 / ctx.model.activation_std / ctx.dims.sizes.heads,
               ctx.dims.features, ctx.dims.inner_bottleneck_features)
    inp = scale_norm_act(ctx, inp, psum=True)
    inp = conv(ctx, inp, ctx.dims.inner_bottleneck_kernel, 1 / ctx.model.activation_std,
               ctx.dims.inner_bottleneck_features, ctx.dims.inner_bottleneck_features)
    inp = scale_norm_act(ctx, inp)
    return conv(ctx, inp, ctx.dims.outer_bottleneck_kernel, 1 / ctx.model.activation_std,
                ctx.dims.inner_bottleneck_features, ctx.dims.features)


def pointwise_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("pointwise")
    inp = scale_norm_act(ctx, inp)
    inp = conv(ctx, inp, ctx.dims.pointwise_kernel, 1 / ctx.model.activation_std, ctx.dims.features,
               ctx.dims.pointwise_features)
    inp = activate(ctx, inp)
    return conv(ctx, inp, ctx.dims.pointwise_kernel, 1 / ctx.model.activation_std, ctx.dims.pointwise_features,
                ctx.dims.features)


def qrnn(ctx: Context, forget: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    dtype = forget.dtype
    for i in range(int(math.log2(ctx.dims.sizes.sequence))):
        x += jnp.concatenate([jnp.zeros((x.shape[0], 2 ** i, x.shape[2])), x[:, :-2 ** i] * forget[:, 2 ** i:]], 1)
        forget *= jnp.concatenate([jnp.ones((x.shape[0], 2 ** i, x.shape[2])), forget[:, :-2 ** i]], 1)
    return x.astype(dtype)


def qrnn_grad(ctx: Context, forget: jnp.ndarray, src: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(fgt: jnp.ndarray, inp: jnp.ndarray):
        dtype = inp.dtype
        out = qrnn(ctx, jax.nn.hard_sigmoid(promote_to(fgt, jnp.float32)), promote_to(inp, jnp.float32))
        out = out.astype(dtype)

        def _grad(dy: jnp.ndarray):
            x = promote_to(inp, jnp.float32)
            f = jax.nn.hard_sigmoid(promote_to(fgt, jnp.float32))
            f = lax.rev(f, (1,))
            f = jnp.concatenate([jnp.ones((x.shape[0], 1, x.shape[2])), f[:, :-1]], 1)
            dy_rev = lax.rev(dy, (1,))
            dx = lax.rev(qrnn(ctx, f, dy_rev), (1,))
            df = dx * promote_to(out, jnp.float32)
            df = jnp.where(jnp.logical_or(fgt > 3, fgt < -3), 0, df / 6)
            df = df.astype(dtype)
            dx = dx.astype(dtype)
            return df, dx

        return out, _grad

    return _fn(forget, src)


def qrnn_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    # 500ms at 256 features (forward pass, backward takes slightly longer)
    # While conv 256->256 with kernel_size=5 takes ~11.3ms
    ctx = ctx.add_to_prefix("qrnn")
    forget = conv(ctx, inp, ctx.dims.pointwise_kernel, 1, ctx.dims.features, ctx.dims.inner_bottleneck_features)
    mid = conv(ctx, inp, ctx.dims.pointwise_kernel, 1, ctx.dims.features, ctx.dims.inner_bottleneck_features)
    out = qrnn_grad(ctx, forget, mid)
    out = scale_norm_act(ctx, out)
    return conv(ctx, out, ctx.dims.pointwise_kernel, 1 / ctx.model.activation_std,
                ctx.dims.inner_bottleneck_features, ctx.dims.features)


def z_loss(ctx: Context, src: jnp.ndarray, use_previous_grad: bool = True) -> jnp.ndarray:
    # forward: 0 (-> to not change loss)
    # backward: grad(jnp.square(log_z).mean() * ctx.training.z_loss)
    @jax.custom_gradient
    def _fn(inp: jnp.ndarray):
        def _grad(dy):
            grad = ctx.training.z_loss / inp.size
            if use_previous_grad:
                grad = grad * dy
            return inp * grad

        return jnp.zeros((), dtype=inp.dtype), _grad

    return _fn(src)


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def top1_gating(ctx: Context, gate: jnp.ndarray, x: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    # prepare shapes
    batch, sequence, experts = gate.shape
    features = x.shape[-1]
    tokens = batch * sequence
    overflow = tokens // experts
    dtype = x.dtype
    gate = promote_to(gate, jnp.float32)
    x = promote_to(x, jnp.float32)
    gate = gate.reshape(batch * sequence, experts)

    # parallel-softmax gate
    max_gate = lax.pmax(lax.stop_gradient(gate), ParallelAxes.model)
    lse = lax.psum(jnp.exp(gate - max_gate), ParallelAxes.model) + max_gate
    lse += z_loss(ctx, lse, False)  # actual zloss
    gate = jnp.exp(gate)
    gate += z_loss(ctx, gate, False)  # aux loss
    balanced = gate / lax.stop_gradient(gate).sum(0)  # balance gates across batch

    # shuffle to avoid imbalances across token position (https://arxiv.org/abs/2109.10465)
    ctx.prng_key, key = jax.random.split(ctx.prng_key)
    indices = jnp.argsort(jax.random.normal(key, (gate.shape[0],)), 0)
    balanced = jnp.take_along_axis(balanced, jnp.broadcast_to(indices.reshape(-1, 1), gate.shape), 0)

    # avoid overflow / get best index
    assignments = jnp.argsort(balanced, -1)
    square_hot = one_hot(assignments, features)
    mask = (square_hot.cumsum(0) > overflow).cumsum(2) < 1
    square_hot = jnp.bitwise_and(square_hot, mask)
    mask = square_hot.sum(-1)
    mask = mask * experts
    assignments = jnp.argsort(assignments, -1)
    assignments = assignments - mask
    assignments = jnp.argmax(assignments, -1)

    # unshuffle
    indices = jnp.argsort(indices)
    assignments = jnp.take_along_axis(assignments, indices, 0)

    # get slice of tokens
    index = lax.psum_scatter(jnp.arange(ctx.dims.sizes.heads), ParallelAxes.model) / ctx.dims.sizes.heads
    own_indices = jnp.argsort(assignments == index)[-overflow:]
    weight = jnp.take_along_axis(gate, assignments.reshape(*assignments.shape, 1), -1)
    weight = jnp.take_along_axis(weight, own_indices.reshape(-1, 1), 0)
    x = x.reshape(batch * sequence, features)
    x = jnp.take_along_axis(x, jnp.broadcast_to(own_indices.reshape(-1, 1), (overflow, features)), 0)
    x = x * weight
    x = x.astype(dtype)

    return x, own_indices


def moe(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("moe")
    inp_wgt = get_param(ctx, "ff_input", [ctx.dims.features, ctx.dims.moe_intermediate],
                        scale=1 / ctx.model.activation_std)
    out_wgt = get_param(ctx, "ff_output", [ctx.dims.moe_intermediate, ctx.dims.features])
    out_wgt = rezero(ctx, out_wgt)

    gates = conv(ctx, inp, ctx.dims.pointwise_kernel, 1, ctx.dims.features, ctx.dims.features)
    mid, indices = top1_gating(ctx, gates, inp)
    mid = matmul(mid, inp_wgt)
    mid = activate(ctx, mid)
    out = matmul(mid, out_wgt)
    return jnp.zeros_like(inp).reshape(-1, inp.shape[-1]).at[indices].set(out).reshape(inp.shape)


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")
    param = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.features], std=1e-5)
    normalization_scale = get_param(ctx, "normalization_scale", [ctx.dims.one], std=0,
                                    dtype=jnp.promote_types(ctx.model.computation_dtype, jnp.float32))

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return scale_norm_act(ctx, jnp.take(wgt, src, 0), scale)

    return jax.checkpoint(_fn)(inp, param, normalization_scale)


def output_embed_shard(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    embd = get_param(ctx, "embd", [ctx.dims.features, ctx.dims.vocab], std=0,
                     lr_scale=1 / ctx.dims.sizes.heads)
    normalization_scale = get_param(ctx, "normalization_scale", [ctx.dims.one], std=0,
                                    dtype=jnp.promote_types(ctx.model.computation_dtype, jnp.float32))
    if ctx.is_initializing:
        return inp

    def _fn(src: jnp.ndarray, wgt: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
        return matmul(scale_norm_act(ctx, src, scale), wgt)

    return jax.checkpoint(_fn)(inp, embd, normalization_scale)


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


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    # Forward: logsumexp(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - 1 + softmax(x) * logsumexp(x) * z_loss
    devices = ctx.dims.sizes.heads
    dtype = src.dtype

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray, inner_tgt: jnp.ndarray):
        inp = inp.reshape(devices, ctx.dims.sizes.batch * ctx.dims.sizes.sequence // devices, ctx.dims.sizes.vocab)
        inp = lax.psum_scatter(inp, ParallelAxes.model).reshape(-1, ctx.dims.sizes.vocab)
        index = lax.psum_scatter(jnp.arange(ctx.dims.sizes.heads), ParallelAxes.model) // devices
        index = index.astype(jnp.int32)
        inner_tgt = lax.dynamic_slice_in_dim(inner_tgt.reshape(-1), index * inp.shape[0], inp.shape[0])
        lse = jax.nn.logsumexp(promote_to(inp, jnp.float32), 1, keepdims=True)

        def _grad(dy: typing.Tuple[jnp.ndarray, None]):
            dy, _ = dy
            dy = promote_to(dy, jnp.float32)
            dy = dy / inner_tgt.size
            tmp = promote_to(inp, jnp.float32)
            dx = lax.exp(tmp - lse)

            zloss = dx * lse * (ctx.training.z_loss * dy)
            dx = dx.at[jnp.arange(dx.shape[0]).reshape(-1, 1), inner_tgt.reshape(-1, 1)].add(-1) * dy
            dx = dx + zloss
            d_src = lax.all_gather(dx, ParallelAxes.model).reshape(src.shape)
            return d_src.astype(dtype), None

        loss = lse - jnp.take_along_axis(inp, inner_tgt.reshape(*inner_tgt.shape, 1), -1)
        accuracy = jnp.argmax(lax.stop_gradient(inp), 1) == inner_tgt
        loss = promote_to(loss, jnp.float32)
        accuracy = promote_to(accuracy, jnp.float32)
        loss = lax.psum(loss.mean() / ctx.dims.sizes.heads, ParallelAxes.model)
        accuracy = lax.psum(accuracy.mean() / ctx.dims.sizes.heads, ParallelAxes.model)
        return (loss, accuracy), _grad

    return _fn(src, tgt)


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
        src = reversible(ctx, pointwise_block, src)
        src = reversible(ctx, bottleneck_block, src)
        src = reversible(ctx, pointwise_block, src)
        # src = reversible(ctx, moe, src)
        # src = reversible(ctx, qrnn_block, src)  # <-- perhaps use it every N blocks? or less features in RNN?
    ctx.parameters = src[0]
    return output_embed_shard(ctx, revnet_out(src[1:]))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return cross_entropy_loss(ctx, body_ctx(ctx, src), tgt)
