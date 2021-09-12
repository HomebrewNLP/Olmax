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


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(src: jnp.ndarray):
        mean = shard(src.mean(-1, keepdims=True))
        out = src - mean
        scale = norm(ctx, out, -1, True) * src.shape[-1] ** -0.5
        out = out * scale

        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            tmp_dy = dy
            tmp_dy *= scale
            tmp_dy -= tmp_dy.mean(-1, keepdims=True)
            normed = out / out.shape[-1]
            tmp_dy -= (dy * normed).sum(-1, keepdims=True) * scale ** 2 * (normed - normed.mean(-1, keepdims=True))
            return tmp_dy

        return out, _grad

    return _fn(inp)


def feed_forward_features(ctx: Context, in_dim: str, out_dim: str) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    inp_weight = get_param(ctx, "inp_weight", [ctx.dims.heads, in_dim, out_dim], scale=1 / ctx.model.activation_std)
    out_weight = get_param(ctx, "out_weight", [ctx.dims.heads, out_dim, in_dim], scale=ctx.model.depth ** -0.5)
    return inp_weight, out_weight


def group_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("group_feed_forward")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.features_per_head, ctx.dims.intermediate_parallel)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(dot(normed, inp_weight, -1, 1, -2, 0), 0, 1))
    out = shard(dot(mid, out_weight, -1, 1, 0, 0), 0, 1)
    out = shard(transpose(out, tuple(range(1, inp.ndim - 1)) + (0, -1)))
    return out


def feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("feed_forward")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.features_per_head, ctx.dims.intermediate_replicated)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(matmul(normed, inp_weight, 2), None))
    out = shard(dot(mid, out_weight, -1, 1))
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


def contrastive_output_embed(ctx: Context, inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx.add_to_prefix("contrastive_output_embed")
    return lax.stop_gradient(inp), group_feed_forward(ctx, inp)


def reversible(ctx: Context, fn: typing.Callable, is_last: bool):
    name_cache = copy.deepcopy(ctx.name_cache)

    if ctx.is_initializing:
        def _fn(inp: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
            params, x00, x01, x10, x11 = inp
            new_ctx = ctx.add_to_prefix("reversible")
            new_ctx.parameters = params
            out = fn(new_ctx, x10)
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
        out = fn(new_ctx, inp)
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
        if is_last:
            y1 = inp[3]
            y0 = inp[1]

        x0, grad_fn = jax.vjp(base, (params, y0))
        d_params, dx0 = grad_fn(dy1)[0]
        d_params = d_params if is_last else {k: d_params_old[k] + d_params[k] for k in d_params.keys()}
        return (d_params, dy1, y1 - x0, dx0 + dy0, y0),

    reversible_half_residual.defvjp(reversible_forward, reversible_backward)
    return reversible_half_residual


def mean_gate(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("mean_gate")
    return feed_forward(ctx, inp.mean(1, keepdims=True))


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


def spatial_mixing(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("spatial_mixing")
    inp_weight, out_weight = feed_forward_features(ctx, ctx.dims.sequence, ctx.dims.sequence)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(dot(normed, inp_weight, -3, 1, -2, 0), 0, 1))  # HBFS
    out = shard(dot(mid, out_weight, -1, 1, 0, 0), 0, 1)
    out = shard(transpose(out, tuple(range(1, inp.ndim - 2)) + (-1, 0, -2)))  # B S H F
    return out


def attention(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    feature_dims = [ctx.dims.heads, ctx.dims.features_per_head]
    base_param = get_param(ctx, "base", feature_dims + [ctx.dims.intermediate_replicated],
                           scale=1 / ctx.model.activation_std)
    key_param = get_param(ctx, "key", [ctx.dims.intermediate_replicated] + feature_dims, column_axes=2)
    qry_param = get_param(ctx, "qry", [ctx.dims.intermediate_replicated] + feature_dims, column_axes=2)
    val_param = get_param(ctx, "val", [ctx.dims.intermediate_replicated] + feature_dims, column_axes=2,
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
    base = activate(ctx, shard(matmul(base, base_param, 2), None))
    key = shard(matmul(base, key_param, 2))
    qry = shard(matmul(base, qry_param, 2))
    val = shard(matmul(base, val_param, 2))

    key = shard(transpose(key, key_permute), -3) * inp.shape[-1] ** -0.5
    val = shard(transpose(val, key_permute), -3)
    qry = shard(transpose(qry, qry_permute), -3)
    lgt = shard(dot(key, qry, feature_dim, head_dim, batch_seq, batch_seq), -3)
    lgt = softmax(ctx, lgt)

    out = shard(dot(lgt, val, feature_dim, head_dim, batch_seq, batch_seq), -3)
    return shard(transpose(out, key_permute))


def contrastive_loss(ctx: Context, out: jnp.ndarray, proj: jnp.ndarray) -> jnp.ndarray:
    """Cosine similarity of output and projection https://arxiv.org/abs/2011.10566"""
    batch_dims = tuple(range(out.ndim - 3))
    out = shard(dot(out, norm(ctx, out, (-2, -1), False, None), -3, -1, batch_dims, batch_dims))
    proj = shard(dot(proj, norm(ctx, proj, (-2, -1), False, None), -3, -1, batch_dims, batch_dims))

    normalization = -1 / ctx.dims.sizes.sequence ** 2
    return shard(dot(out, proj, (-2, -1), (-2, -1), batch_dims, batch_dims), None) * normalization


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> jnp.ndarray:
    normalization = ctx.dims.sizes.batch / tgt.size
    tgt = shard(one_hot(tgt.astype(src.dtype), src.shape[-1]), None)
    shifted = src - shard(src.max(axis=-1, keepdims=True), None)
    exp_shifted = jnp.exp(shifted)
    sum_exp = shard(jnp.sum(exp_shifted, axis=-1, keepdims=True), None)
    return shard(((jnp.log(sum_exp) - shifted) * tgt).sum(tuple(range(1, tgt.ndim))), None) * normalization


def body_ctx(ctx: Context, src: jnp.ndarray) -> typing.Union[typing.Tuple[jnp.ndarray, jnp.ndarray], jnp.ndarray]:
    src = input_embed(ctx, src)
    zero = shard(jnp.zeros_like(src))
    src = (ctx.parameters, src, zero, src, zero)
    for i in range(ctx.model.depth):
        src = reversible(ctx, spatial_mixing, (i + 1) == ctx.model.depth)(src)
        src = reversible(ctx, feed_forward, (i + 1) == ctx.model.depth)(src)
    src = src[1] + src[3]
    return contrastive_output_embed(ctx, src) if ctx.training.contrastive else output_embed(ctx, src)


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    if ctx.training.contrastive:
        unreduced_loss = contrastive_loss(ctx, *body_ctx(ctx, shard(inp, None)))
    else:
        src, tgt = inp
        unreduced_loss = cross_entropy_loss(ctx, body_ctx(ctx, shard(src, None)), shard(tgt, None))
    top_loss = loss = unreduced_loss.sum() / ctx.dims.sizes.batch
    top_k = math.ceil(ctx.dims.sizes.batch * ctx.training.loss_top_p / ctx.training.loss_top_snap)
    top_k *= ctx.training.loss_top_snap
    if ctx.training.loss_top_p < 1 and top_k < ctx.dims.sizes.batch:
        top_loss, _ = lax.top_k(unreduced_loss, top_k)
        top_loss = top_loss.sum() / top_k
    return top_loss, loss
