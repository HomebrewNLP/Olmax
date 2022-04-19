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
        return jnp.zeros(inp.shape[:-1] + (ctx.dims.sizes[out_features],))
    return lax_conv(inp, weight, [(weight.shape[-1] - 1, 0)], ctx.dims.sizes[out_features] if depthwise else 1)


def full_conv(ctx: Context, inp: jnp.ndarray, scale: float, in_features: str, out_features: str,
              use_rezero: bool = False) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("full_conv")
    return conv(ctx, inp, False, ctx.dims.full_conv_kernel, scale, in_features, out_features, use_rezero)


def depthwise_conv(ctx: Context, inp: jnp.ndarray, scale: float, out_features: str,
                   use_rezero: bool = False) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("depthwise_conv")
    return conv(ctx, inp, True, ctx.dims.depthwise_conv_kernel, scale, ctx.dims.one, out_features, use_rezero)


def output_conv(ctx: Context, inp: jnp.ndarray, in_features: typing.Optional[str] = None):
    ctx = ctx.add_to_prefix("output_conv")
    return full_conv(ctx, inp, ctx.dims.sizes.depth ** -0.5,
                     ctx.dims.intermediate if in_features is None else in_features,
                     ctx.dims.features_per_head, True)


def depthwise_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("depthwise")
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


def reduced_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("reduced")
    mid = full_conv(ctx, inp, 1 / ctx.model.activation_std / ctx.dims.sizes.heads, ctx.dims.features_per_head,
                    ctx.dims.intermediate)
    mid = activated_allsum(ctx, mid)
    return output_conv(ctx, mid)


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
    ctx = ctx.add_to_prefix("qrnn")
    forget = full_conv(ctx, inp, 1, ctx.dims.features_per_head, ctx.dims.features_per_head)
    mid = full_conv(ctx, inp, 1, ctx.dims.features_per_head, ctx.dims.features_per_head)
    out = qrnn(ctx, forget, mid)
    out = scale_norm(ctx, out)
    return output_conv(ctx, out, ctx.dims.features_per_head)


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
    inp_wgt = get_param(ctx, "ff_input", [ctx.dims.features_per_head, ctx.dims.moe_intermediate],
                        scale=1 / ctx.model.activation_std)
    out_wgt = get_param(ctx, "ff_output", [ctx.dims.moe_intermediate, ctx.dims.features_per_head])
    out_wgt = rezero(ctx, out_wgt, ctx.dims.sizes.depth ** -0.5)

    gates = full_conv(ctx, inp, 1, ctx.dims.features_per_head, ctx.dims.heads)
    mid, indices = top1_gating(ctx, gates, inp)
    mid = matmul(mid, inp_wgt)
    mid = activate(ctx, mid)
    out = matmul(mid, out_wgt)
    return jnp.zeros_like(inp).reshape(-1, inp.shape[-1]).at[indices].set(out).reshape(inp.shape)


def reduced_self_conv_block(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    # Uses 4x padding
    sequence = ctx.dims.sizes.sequence
    features = ctx.dims.sizes.features_per_head * ctx.dims.sizes.batch
    weight = full_conv(ctx, inp, 1, ctx.dims.features_per_head, ctx.dims.features_per_head)
    inp = inp.transpose(1, 0, 2).reshape(1, sequence, features)
    weight = weight.transpose(0, 2, 1).reshape(features, 1, sequence)
    weight = jnp.flip(weight, 2)
    mid = lax_conv(inp, weight, [(sequence - 1, 0)], features)
    mid = mid.reshape(sequence, ctx.dims.sizes.batch, ctx.dims.sizes.features_per_head).transpose(1, 0, 2)
    mid = scale_norm(ctx, mid)
    mid = activated_allsum(ctx, mid)
    return output_conv(ctx, mid, ctx.dims.features_per_head)


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


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    # Forward: max(x) - x[target]
    # Backward: (logsumexp(x) - x[target] + logsumexp(x)^2 * z_loss).grad
    # -> softmax(x) - 1 + softmax(x) * logsumexp(x) * z_loss
    devices = ctx.dims.sizes.heads

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray, inner_tgt: jnp.ndarray):
        inp = inp.reshape(devices, ctx.dims.sizes.batch * ctx.dims.sizes.sequence // devices, ctx.dims.sizes.vocab)
        inp = lax.psum_scatter(inp, ParallelAxes.model).reshape(-1, ctx.dims.sizes.vocab)
        index = lax.psum_scatter(jnp.arange(ctx.dims.sizes.heads), ParallelAxes.model) // devices
        index = index.astype(jnp.int32)
        inner_tgt = inner_tgt[index * inp.shape[0] // devices:(index + 1) * inp.shape[0] // devices]
        inner_tgt = inner_tgt.reshape(-1, ctx.dims.sizes.vocab)

        def _grad(dy: typing.Tuple[jnp.ndarray, None]):
            dy, _ = dy
            dy = promote_to(dy, jnp.float32)
            dy = dy / (src.size / ctx.dims.vocab)
            tmp = promote_to(inp, jnp.float32)
            lse = jax.nn.logsumexp(tmp)
            dx = lax.exp(tmp - lse)

            zloss = dx * lse * (ctx.training.z_loss * dy)
            dx = dx.at[inner_tgt].add(-1) * dy
            dx = dx + zloss
            d_src = lax.all_gather(dx, ParallelAxes.model).reshape(src.shape)
            return d_src, None

        loss = inp.max(-1) - jnp.take_along_axis(inp, inner_tgt.reshape(*inner_tgt.shape, 1), -1)
        accuracy = jnp.argmax(lax.stop_gradient(inp), 2) == inner_tgt
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
        src = reversible(ctx, reduced_block, src)
        src = reversible(ctx, depthwise_block, src)
        # src = reversible(ctx, reduced_self_conv_block, src)
        # src = reversible(ctx, moe, src)
        src = reversible(ctx, qrnn_block, src)
    ctx.parameters = src[0]
    return output_embed_shard(ctx, revnet_out(src[1:]))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return cross_entropy_loss(ctx, body_ctx(ctx, src), tgt)
