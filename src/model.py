import copy
import math
import typing

import jax
import jax._src.util as util
from jax import lax, numpy as jnp

from src.backend import get_param, shard, dims_to_shape, INT_OR_TUPLE, dot, matmul, transpose
from src.context import Context

REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def activate(ctx, inp: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.leaky_relu(inp, ctx.model.leaky_relu_slope)


def norm(ctx: Context, inp: jnp.ndarray, dims: INT_OR_TUPLE, keepdims=False,
         model_parallel_dim: typing.Optional[int] = -2, data_parallel_dim: typing.Optional[int] = 0) -> jnp.ndarray:
    square = shard(jnp.square(inp).sum(dims, keepdims=keepdims), model_parallel_dim, data_parallel_dim)
    return lax.rsqrt(ctx.model.norm_eps + square)


def get_item(inp: jnp.ndarray, idx: int) -> jnp.ndarray:
    pad = (1,) * inp.ndim
    return lax.slice(inp, (idx,) + pad, (idx + 1,) + pad)


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(src: jnp.ndarray):
        mean = shard(src.mean(-1, keepdims=True))
        out = src - mean
        scale = norm(ctx, out, -1, True) * src.shape[-1] ** -0.5
        out = out * scale

        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            tmp_dy = dy * scale
            tmp_dy -= tmp_dy.mean(-1, keepdims=True)
            normed = out / out.shape[-1]
            tmp_dy += (dy * normed).sum(-1, keepdims=True) * scale ** 2 * (normed.mean(-1, keepdims=True) - normed)
            return tmp_dy

        return out, _grad

    return _fn(inp)


def feed_forward_features(ctx: Context, in_dim: str, out_dim: str) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    inp_weight = get_param(ctx, "inp_weight", [ctx.dims.depth, ctx.dims.heads, in_dim, out_dim],
                           scale=1 / ctx.model.activation_std)
    out_weight = get_param(ctx, "out_weight", [ctx.dims.depth, ctx.dims.heads, out_dim, in_dim],
                           scale=ctx.model.depth ** -0.5)
    return inp_weight, out_weight


def group_feed_forward(ctx: Context, inp: jnp.ndarray, idx: int) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("group_feed_forward")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.features_per_head, ctx.dims.intermediate_parallel)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(dot(normed, get_item(inp_weight, idx), -1, 1, -2, 0), 0, 1))
    out = shard(dot(mid, get_item(out_weight, idx), -1, 1, 0, 0), 0, 1)
    out = shard(transpose(out, tuple(range(1, inp.ndim - 1)) + (0, -1)))
    return out


def feed_forward(ctx: Context, inp: jnp.ndarray, idx: int) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("feed_forward")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.features_per_head, ctx.dims.intermediate_replicated)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(matmul(normed, get_item(inp_weight, idx), 2), None))
    out = shard(dot(mid, get_item(out_weight, idx), -1, 1))
    return out


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")

    inp_embd = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.heads, ctx.dims.features_per_head])
    if ctx.is_initializing:
        return jnp.zeros([1] * (inp.ndim + 1))

    out = shard(matmul(one_hot(inp, ctx.data.vocab_size).astype(ctx.model.dtype), inp_embd))
    position_shape = dims_to_shape(ctx, [ctx.dims.sequence])
    feature_shape = dims_to_shape(ctx, [ctx.dims.heads, ctx.dims.features_per_head])
    position_count = util.prod(position_shape)
    feature_count = util.prod(feature_shape)
    positions = jnp.reshape(jnp.arange(0, position_shape), (-1, 1, 1))
    features = jnp.arange(0, feature_count)
    features = shard(jnp.reshape(features, [1] + feature_shape) * 4 / feature_count, 1, None)
    features = jnp.exp(shard(features - math.log(position_count / 2 / math.pi), 1))
    pos_embd = jnp.sin(features * positions).astype(ctx.model.dtype)
    return out + lax.stop_gradient(pos_embd)


def output_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    embd = get_param(ctx, "weight", [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.vocab], 0, 0)
    if ctx.is_initializing:
        return inp
    return shard(matmul(inp, embd, 2), None)


def reversible(ctx: Context, fn: typing.Callable, idx: int):
    name_cache = copy.deepcopy(ctx.name_cache)

    if True:  # ctx.is_initializing:
        def _fn(inp: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
            params, x00, x01, x10, x11 = inp
            new_ctx = ctx.add_to_prefix("reversible")
            new_ctx.parameters = params
            out = fn(new_ctx, x10, idx)
            ctx.parameters = new_ctx.parameters
            ctx.parameter_dims = new_ctx.parameter_dims
            ctx.name_cache = new_ctx.name_cache
            ctx.prng_key = new_ctx.prng_key
            return new_ctx.parameters, x10, x11, out, x01

        return _fn

    def base(inp: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        params, inp = inp
        ctx.name_cache = copy.deepcopy(name_cache)
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, inp, idx)
        ctx.name_cache = new_ctx.name_cache
        return out

    @jax.custom_vjp
    def reversible_half_residual(inp: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
        params, x00, x01, x10, x11 = inp
        out = base((params, x10)) + x00
        return params, x10, x10, out, out

    def reversible_forward(inp: REVERSIBLE_CTX) -> typing.Tuple[REVERSIBLE_CTX, REVERSIBLE_CTX]:
        out = reversible_half_residual(inp)
        return out, out

    def reversible_backward(inp: REVERSIBLE_CTX, dy: REVERSIBLE_CTX) -> typing.Tuple[REVERSIBLE_CTX]:
        d_params_old, dy0, y0, dy1, y1 = dy
        params = inp[0]
        x0, grad_fn = jax.vjp(base, (params, y0))
        d_params, dx0 = grad_fn(dy1)[0]
        d_params = {k: d_params_old.get(k, 0) + d_params.get(k, 0) for k in d_params.keys()}
        return (d_params, dy1, y1 - x0, dx0 + dy0, y0),

    reversible_half_residual.defvjp(reversible_forward, reversible_backward)
    return reversible_half_residual


def softmax(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(lgt: jnp.ndarray):
        if ctx.model.masked_attention:
            ones = (1,) * (lgt.ndim - 2)
            arange = jnp.arange(0, lgt.shape[-1])
            mask: jnp.ndarray = jnp.greater(jnp.reshape(arange, ones + (1, -1)), jnp.reshape(arange, ones + (-1, 1)))
            lgt += (-1e30 * mask).astype(lgt.dtype)
        lgt = jnp.exp(lgt - shard(lgt.max(-1, keepdims=True), -3))
        lgt /= shard(lgt.sum(-1, keepdims=True), -3)

        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            prod = lgt * dy
            return prod - shard(prod.sum(-1, keepdims=True), -3) * lgt

        return lgt, _grad

    return _fn(inp)


def spatial_mixing(ctx: Context, inp: jnp.ndarray, idx: int) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("spatial_mixing")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.sequence, ctx.dims.sequence)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(dot(normed, get_item(inp_weight, idx), -3, 1, -2, 0), 0, 1))  # HBFS
    out = shard(dot(mid, get_item(out_weight, idx), -1, 1, 0, 0), 0, 1)
    out = shard(transpose(out, tuple(range(1, inp.ndim - 2)) + (-1, 0, -2)))  # B S H F
    return out


def attention(ctx: Context, inp: jnp.ndarray, idx: int) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    feature_dims = [ctx.dims.heads, ctx.dims.features_per_head]
    base_param = get_param(ctx, "base", [ctx.dims.depth] + feature_dims + [ctx.dims.intermediate_replicated],
                           scale=1 / ctx.model.activation_std)
    key_param = get_param(ctx, "key", [ctx.dims.depth, ctx.dims.intermediate_replicated] + feature_dims, column_axes=2)
    qry_param = get_param(ctx, "qry", [ctx.dims.depth, ctx.dims.intermediate_replicated] + feature_dims, column_axes=2)
    val_param = get_param(ctx, "val", [ctx.dims.depth, ctx.dims.intermediate_replicated] + feature_dims, column_axes=2,
                          scale=ctx.model.depth ** -0.5)
    if ctx.is_initializing:
        return inp

    batch_dims = tuple(range(inp.ndim - 3))
    head_dim = inp.ndim - 2
    feature_dim = inp.ndim - 1
    sequence_dim = inp.ndim - 3

    key_permute = batch_dims + (head_dim, sequence_dim, feature_dim)
    qry_permute = batch_dims + (head_dim, feature_dim, sequence_dim)
    batch_seq = batch_dims + (sequence_dim,)

    base = instance_norm(ctx, inp)
    base = activate(ctx, shard(matmul(base, get_item(base_param, idx), 2), None))
    key = shard(matmul(base, get_item(key_param, idx), 2))
    qry = shard(matmul(base, get_item(qry_param, idx), 2))
    val = shard(matmul(base, get_item(val_param, idx), 2))

    key = shard(transpose(key, key_permute), -3) * inp.shape[-1] ** -0.5
    val = shard(transpose(val, key_permute), -3)
    qry = shard(transpose(qry, qry_permute), -3)
    lgt = shard(dot(key, qry, feature_dim, head_dim, batch_seq, batch_seq), -3)
    lgt = softmax(ctx, lgt)

    out = shard(dot(lgt, val, feature_dim, head_dim, batch_seq, batch_seq), -3)
    return shard(transpose(out, key_permute))


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> jnp.ndarray:
    normalization = ctx.dims.sizes.batch / tgt.size
    tgt = shard(one_hot(tgt.astype(src.dtype), src.shape[-1]), None)
    shifted = src - shard(src.max(-1, keepdims=True), None)
    exp_shifted = jnp.exp(shifted)
    sum_exp = shard(exp_shifted.sum(-1, keepdims=True), None)
    return shard(((jnp.log(sum_exp) - shifted) * tgt).sum(tuple(range(1, tgt.ndim))), None) * normalization


def momentumnet_main(ctx: Context, fn: typing.Callable):
    def _fn(*x: jnp.ndarray) -> jnp.ndarray:
        *x, idx = x
        return fn(*x, idx) * (1 - ctx.model.momentumnet_beta) / (ctx.model.momentumnet_beta ** idx)

    return _fn


def momentumnet_side(ctx):
    def _fn(_ignored, x: jnp.ndarray, idx: int) -> jnp.ndarray:
        return x * ctx.model.momentumnet_beta ** idx

    return _fn


def step(ctx: Context):
    side = momentumnet_side(ctx)
    if not ctx.is_initializing:
        ctx.parameters = {}

    def _fn(idx: int, src: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
        if not ctx.is_initializing:
            ctx.parameters = {}
        src = reversible(ctx, momentumnet_main(ctx, spatial_mixing), idx)(src)
        src = reversible(ctx, side, idx)(src)
        src = reversible(ctx, momentumnet_main(ctx, feed_forward), idx)(src)
        src = reversible(ctx, side, idx)(src)
        if not ctx.is_initializing:
            ctx.parameters = {}
        return src

    return _fn


def revnet_out(src: typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(x0: jnp.ndarray, x0_back: jnp.ndarray, x1: jnp.ndarray, x1_back: jnp.ndarray):
        def _grad(dy) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            return dy, x0, dy, x1

        return x0 + x1, _grad

    return _fn(*src)


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = shard(jnp.zeros_like(src))
    src = (ctx.parameters, src, zero, src, zero)
    ctx.parameters = {}
    if ctx.is_initializing:
        src = step(ctx)(0, src)
    else:
        for i in range(ctx.dims.sizes.depth):
            src = step(ctx)(i, src)
    ctx.parameters = src[0]
    return output_embed(ctx, revnet_out(src[1:]))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    unreduced_loss = cross_entropy_loss(ctx, body_ctx(ctx, shard(src, None)), shard(tgt, None))
    top_loss = loss = unreduced_loss.sum() / ctx.dims.sizes.batch
    top_k = math.ceil(ctx.dims.sizes.batch * ctx.training.loss_top_p / ctx.training.loss_top_snap)
    top_k *= ctx.training.loss_top_snap
    if ctx.training.loss_top_p < 1 and top_k < ctx.dims.sizes.batch:
        top_loss, _ = lax.top_k(unreduced_loss, top_k)
        top_loss = top_loss.sum() / top_k
    return top_loss, loss
