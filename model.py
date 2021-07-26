import copy
import math
import time
import typing
import warnings

import jax
import jax._src.util as util
import numpy as np
from jax import lax, numpy as jnp, random
from jax.experimental import PartitionSpec
from jax.experimental import pjit
from jax.experimental.maps import mesh
from jax.scipy.special import expit as sigmoid

from context import Context, WhileTrainContext
from data import text_dataset

warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")


# jax.config.update("jax_disable_jit", True)

def dims_to_shape(ctx: Context, dims: typing.List[str]) -> typing.List[int]:
    return [ctx.dims.sizes[d] for d in dims]


def is_intermediate(ctx, inp: jnp.ndarray) -> bool:
    return inp.shape[-1] != ctx.dims.sizes.features_per_head


def get_feature_dim(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    return ctx.dims.intermediate_feed_forward if is_intermediate(ctx, inp) else ctx.dims.features_per_head


def shard(tensor: jnp.ndarray, head: typing.Optional[int] = -2, batch: typing.Optional[int] = 0):
    spec: typing.List[typing.Optional[str]] = [None] * tensor.ndim
    if isinstance(batch, int):
        spec[batch] = "data_parallel"
    if isinstance(head, int):
        spec[head] = "model_parallel"
    try:
        return pjit.with_sharding_constraint(tensor, PartitionSpec(*spec))
    except ValueError as e:
        e_str = str(e)
        if ("One of with_sharding_constraint arguments was given the resource assignment of PartitionSpec(" in e_str and
                ", but resource axis " in e_str and "is undefined. Did you forget to declare the mesh?" in e_str):
            return tensor
        raise e


def default(value: typing.Any, default_value: typing.Any) -> typing.Any:
    return default_value if value is None else value


def dot_general(left: jnp.ndarray, right: jnp.ndarray, left_contract_dims: typing.Sequence[int],
                right_contract_dims: typing.Sequence[int], left_batch_dims: typing.Sequence[int] = tuple(),
                right_batch_dims: typing.Sequence[int] = tuple()) -> jnp.ndarray:
    dims = ((left_contract_dims, right_contract_dims), (left_batch_dims, right_batch_dims))
    return lax.dot_general(left, right, dims, "fastest")


def dot_product(left: jnp.ndarray, right: jnp.ndarray, left_sum_start: int, right_sum_start: int,
                left_sum_end: typing.Optional[int] = None, right_sum_end: typing.Optional[int] = None) -> jnp.ndarray:
    """
    Performs a basic dot product across two tensors. All dimensions in front of left_sum_start and behind right_sum_end
    are kept. All dimensions in front of right_sum_start are batch dimensions. left_sum_end has to be the true end.
    While _start is inclusive, _end is inclusive. Example: 1, 3 gives (1, 2, 3), as it is shifted internally.
    If _end is not set, _start is used as the only value (equal to setting end to start + 1)

    :param left: tensor that's dot-product-ed. left in left@right
    :param right: right tensor from left@right
    :param left_sum_start: start/only summed dim in left tensor
    :param right_sum_start: stort/only summed dim in right tensor
    :param left_sum_end: optional end if multiple dims are used
    :param right_sum_end: optional end if multiple dims are used
    :return: tensor containing left@right
    """
    l_ndim = left.ndim
    r_ndim = right.ndim
    l_start = left_sum_start % l_ndim
    r_start = right_sum_start % r_ndim
    l_end = default(left_sum_end, left_sum_start) % l_ndim + 1
    r_end = default(right_sum_end, right_sum_start) % r_ndim + 1
    min_start = min(r_start, l_start)
    return dot_general(left, right, tuple(range(l_start, l_end)), tuple(range(r_start, r_end)),
                       tuple(range(l_ndim - l_start, l_ndim - l_start - min_start, -1)),
                       tuple(range(r_ndim - r_start, r_ndim - r_start - min_start, -1)))


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axis=-1) -> jnp.ndarray:
    n_rows, n_cols = util.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.model.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim) * ctx.model.initializer.scale
    if n_rows < n_cols:
        out = out.T
    return jnp.moveaxis(jnp.reshape(out, tuple(np.delete(shape, column_axis)) + (shape[column_axis],)), -1, column_axis)


def get_param(ctx: Context, name: str, shape: typing.Optional[typing.List[str]] = None,
              std: typing.Optional[float] = None, mean: typing.Optional[float] = None,
              column_axis: typing.Optional[int] = None) -> jnp.ndarray:
    name = ctx.add_to_prefix(name, count=False).global_prefix
    if name not in ctx.parameters:
        ctx.parameter_dims[name] = shape
        shape = dims_to_shape(ctx, shape)
        if std is None and mean is None:
            ctx.parameters[name] = orthogonal_init(ctx, shape, -1 if column_axis is None else column_axis)
        else:
            ctx.parameters[name] = random.normal(ctx.prng_key, shape, ctx.model.dtype)
            if std is not None:
                ctx.parameters[name] *= std
            if mean is not None:
                ctx.parameters[name] += mean
    return ctx.parameters[name]


def zero_param(ctx: Context, name: str, shape: typing.List[str]) -> jnp.ndarray:
    return get_param(ctx, name, shape, 0, 0)


def one_shape(ndim: int, dim_name: str, dim_idx: int) -> typing.List[str]:
    base = ["one"] * ndim
    base[dim_idx] = dim_name
    return base


def optimizer_rsqrt(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.reciprocal(jnp.maximum(jnp.sqrt(inp), 1e-5))


def sm3(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("sm3", count=False)
    dims = ctx.parameter_dims[param_name] if param_name in ctx.parameter_dims else ["one"] * grad.ndim
    weight_update = zero_param(ctx, "dim0", one_shape(grad.ndim, dims[0], 0))
    buffer = [weight_update]
    head_index = dims.index(ctx.dims.heads) if ctx.dims.heads in dims else -1

    for i, d in enumerate(dims[1:], 1):
        buffer.append(zero_param(ctx, f"dim{i}", one_shape(grad.ndim, d, i)))
        weight_update = jnp.minimum(weight_update, buffer[-1])

        if i >= head_index >= 0:
            weight_update = shard(weight_update, head_index, None)

    weight_update = weight_update + jnp.square(grad)

    for i in range(grad.ndim):
        new = weight_update.max([j for j in range(grad.ndim) if j != i], keepdims=True)
        ctx.parameters[ctx.add_to_prefix(f"dim{i}", count=False).global_prefix] = new

    return grad * optimizer_rsqrt(weight_update)


def adaptive_gradient_clipping(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    grd_norm = jnp.maximum(jnp.sqrt(jnp.square(grad).sum()), 1e-6)
    wgt_norm = jnp.maximum(jnp.sqrt(jnp.square(ctx.parameters[param_name]).sum()), 1e-3)
    do_clip = jnp.greater(grd_norm * jnp.reciprocal(wgt_norm), ctx.optimizer.gradient_clip)
    clipped = wgt_norm * jnp.reciprocal(grd_norm) * ctx.optimizer.gradient_clip * grad
    return clipped * do_clip + grad * (1 - do_clip)


def momentum(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("momentum", count=False)
    state = zero_param(ctx, "momentum_buffer", ctx.parameter_dims.get(param_name))
    new_state = ctx.optimizer.momentum_beta * state + grad
    ctx.parameters[ctx.add_to_prefix("momentum_buffer", count=False).global_prefix] = new_state
    if not ctx.optimizer.nesterov_momentum:
        return new_state
    return grad + ctx.optimizer.momentum_beta * new_state


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))


def relu(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(inp, 0)


def group_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("group_feed_forward")
    ndim = inp.ndim
    features = [ctx.dims.features_per_head, ctx.dims.intermediate_feed_forward]
    inp_weight = get_param(ctx, "inp_weight", [ctx.dims.heads] + features)
    out_weight = get_param(ctx, "out_weight", [ctx.dims.heads] + features[::-1])
    if ctx.is_initializing:
        return inp

    transpose = tuple(range(1, ndim - 1)) + (0, ndim - 1)
    batch_seq = tuple(range(ndim - 2))
    batch_seq_1 = tuple(range(1, ndim - 1))

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, i_w: jnp.ndarray, o_w: jnp.ndarray):
        mid = shard(dot_general(src, i_w, (ndim - 1,), (1,), (ndim - 2,), (0,)), 0, 1)
        out = shard(dot_general(relu(mid), o_w, (ndim - 1,), (1,), (0,), (0,)), 0, 1)
        out = shard(out.transpose(transpose))

        def _grad_fn(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            r_mid = relu(mid)
            o_w_grad = dot_general(r_mid, dy, batch_seq_1, batch_seq, (0,), (2,))
            d_mid = dot_general(dy, o_w, (ndim - 1,), (2,), (ndim - 2,), (0,))
            d_mid = d_mid * jnp.greater(mid, 0).astype(ctx.model.dtype)
            i_w_grad = dot_general(src, d_mid, batch_seq, batch_seq_1, (ndim - 2,), (0,))
            d_src = dot_general(d_mid, i_w, (d_mid.ndim - 1,), (2,), (0,), (0,))
            d_src = d_src.transpose(transpose)
            return d_src, i_w_grad, o_w_grad

        return out, _grad_fn

    return _fn(inp, inp_weight, out_weight)


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")

    inp_embd = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.intermediate_feed_forward],
                         ctx.model.initializer.embedding_std)
    out_embd = get_param(ctx, "out_embd",
                         [ctx.dims.intermediate_feed_forward, ctx.dims.heads, ctx.dims.features_per_head])
    if ctx.is_initializing:
        return jnp.zeros([1] * (inp.ndim + 1))

    out = shard(dot_product(one_hot(inp, ctx.data.vocab_size).astype(ctx.model.dtype), inp_embd, -1, 0), None)
    out = shard(dot_product(relu(out), out_embd, -1, 0))

    position_shape = dims_to_shape(ctx, [ctx.dims.sequence])
    feature_shape = dims_to_shape(ctx, [ctx.dims.heads, ctx.dims.features_per_head])
    position_count = util.prod(position_shape)
    feature_count = util.prod(feature_shape)
    positions = jnp.reshape(jnp.arange(0, position_shape), (-1, 1, 1))
    features = jnp.arange(0, feature_count)
    features = shard(jnp.reshape(features, [1] + feature_shape) * 4 / feature_count, 1, None)
    features = jnp.exp(shard(features - math.log(position_count / 2 / math.pi), 1))
    pos_embd = lax.stop_gradient(jnp.sin(features * positions) * ctx.model.initializer.embedding_std).astype(out.dtype)
    return out + pos_embd


def output_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    embd = get_param(ctx, "weight", [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.vocab])
    if ctx.is_initializing:
        return inp
    return shard(dot_product(inp, embd, -2, 0, -1, 1), None)


REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def reversible(ctx: Context, fn: typing.Callable, is_last: bool):
    name_cache = copy.deepcopy(ctx.name_cache)

    def base(inp: typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray]) -> jnp.ndarray:
        params, inp = inp
        ctx.name_cache = copy.deepcopy(name_cache)
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, inp)
        ctx.name_cache = new_ctx.name_cache
        return out

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

    @jax.custom_vjp
    def reversible_half_residual(inp: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
        params, x00, x01, x10, x11 = inp
        out = base((params, x10)) + x00
        return params, x10, x10, out, out

    def reversible_forward(inp: REVERSIBLE_CTX) -> typing.Tuple[REVERSIBLE_CTX, REVERSIBLE_CTX]:
        out = reversible_half_residual(inp)
        return out, out

    def reversible_backward(inp: REVERSIBLE_CTX, dy: REVERSIBLE_CTX) -> typing.Tuple[REVERSIBLE_CTX]:
        d_params_old, dx10, x10, dy00, y00 = dy
        params = inp[0]
        if is_last:
            y00 = inp[4]
            x10 = inp[2]
        x00 = y00 - base((params, x10))
        _, grad_fn = jax.vjp(base, (params, x10))
        d_params, dx00 = grad_fn(dy00)[0]
        d_params = {k: d_params_old.get(k, 0) + d_params.get(k, 0) for k in d_params.keys()}
        return (d_params, dy00, x00, dx00 + dx10, x10),

    reversible_half_residual.defvjp(reversible_forward, reversible_backward)
    return reversible_half_residual


def attention(ctx: Context, src: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    feature_dims = [ctx.dims.heads, ctx.dims.features_per_head]
    base_param = get_param(ctx, "base", feature_dims + [ctx.dims.intermediate_feed_forward])
    attn_params = [get_param(ctx, name, [ctx.dims.intermediate_feed_forward] + feature_dims)
                   for name in ("key", "query", "value")]
    if ctx.is_initializing:
        return src

    batch_dims = tuple(range(src.ndim - 3))
    head_dim = src.ndim - 2
    feature_dim = src.ndim - 1
    sequence_dim = src.ndim - 3

    key_permute = batch_dims + (head_dim, sequence_dim, feature_dim)
    qry_permute = batch_dims + (head_dim, feature_dim, sequence_dim)
    batch_seq = batch_dims + (sequence_dim,)

    @jax.custom_gradient
    def _fn(inp: jnp.ndarray, b_p: jnp.ndarray, k_p: jnp.ndarray, q_p: jnp.ndarray, v_p: jnp.ndarray):
        base = relu(shard(dot_product(inp, b_p, -2, 0, -1, 1), None))  # batch, seq, feat
        key = shard(dot_product(base, k_p, -1, 0))
        qry = shard(dot_product(base, q_p, -1, 0))
        val = shard(dot_product(base, v_p, -1, 0))

        key = shard(key.transpose(key_permute), -3)
        val = shard(val.transpose(key_permute), -3)
        qry = shard(qry.transpose(qry_permute), -3)

        lgt = shard(dot_general(key, qry, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)
        if ctx.model.masked_attention:
            ones = (1,) * (lgt.ndim - 2)
            arange = jnp.arange(0, lgt.shape[-1])
            mask: jnp.ndarray = jnp.greater(jnp.reshape(arange, ones + (1, -1)), jnp.reshape(arange, ones + (-1, 1)))
            lgt += (-1e30 * mask).astype(lgt.dtype)
        lgt = jnp.exp(lgt - lgt.max(-1, keepdims=True))
        lgt /= lgt.sum(-1, keepdims=True)
        out = shard(dot_general(lgt, val, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)
        out = shard(out.transpose(key_permute))

        def grad_fn(dy: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            dy = shard(dy.transpose(qry_permute), -3)

            val_grad = shard(dot_general(dy, lgt, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)

            d_logit = shard(dot_general(val, dy, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)
            prod = lgt * d_logit
            lgt_grad = prod - shard(prod.sum(-1, keepdims=True), -3) * lgt

            qry_grad = shard(dot_general(lgt_grad, qry, (feature_dim,), (feature_dim,), batch_seq, batch_seq), -3)
            key_grad = shard(dot_general(lgt_grad, key, (head_dim,), (head_dim,), batch_seq, batch_seq), -3)

            k_p_grad = shard(dot_general(base, key_grad, batch_seq, batch_dims + (head_dim,)), 1, None)
            q_p_grad = shard(dot_general(base, qry_grad, batch_seq, batch_dims + (head_dim,)), 1, None)
            v_p_grad = shard(dot_general(base, val_grad, batch_seq, batch_dims + (feature_dim,)), 1, None)

            base_grad = shard(dot_general(key_grad, k_p, (sequence_dim, feature_dim), (1, 2)), None)
            base_grad += shard(dot_general(qry_grad, q_p, (sequence_dim, feature_dim), (1, 2)), None)
            base_grad += shard(dot_general(val_grad, v_p, (sequence_dim, head_dim), (1, 2)), None)

            base_grad = base_grad * jnp.greater(base, 0).astype(ctx.model.dtype)
            inp_grad = shard(dot_general(base_grad, b_p, (head_dim,), (2,)))
            b_p_grad = shard(dot_general(inp, base_grad, batch_seq, batch_seq), 0, None)

            return inp_grad, b_p_grad, k_p_grad, q_p_grad, v_p_grad

        return out, grad_fn

    return shard(_fn(src, base_param, *attn_params))


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("instance_norm")
    shape = ["one"] * (inp.ndim - 2) + [ctx.dims.heads, ctx.dims.features_per_head]
    scale = get_param(ctx, "scale", shape, ctx.model.initializer.norm_std, 1)
    shift = get_param(ctx, "shift", shape, ctx.model.initializer.norm_std)
    if ctx.is_initializing:
        return inp

    inp = inp - shard(inp.mean(-1, keepdims=True), None)
    inp = inp * scale * lax.rsqrt(ctx.model.norm_eps + shard(jnp.square(inp).mean(-1, keepdims=True), None))
    return inp + shift


def exec_fn(*fns: typing.Callable) -> typing.Callable:
    def _run(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
        for f in fns:
            inp = f(ctx, inp)
        return inp

    return _run


@jax.custom_gradient
def cross_entropy_loss(src: jnp.ndarray, tgt: jnp.ndarray):
    spec = base_spec(src)
    normalization = tgt.size
    tgt = one_hot(tgt.astype(src.dtype), src.shape[-1])
    shifted = src - shard(src.max(axis=-1, keepdims=True), None)
    exp_shifted = jnp.exp(shifted)
    sum_exp = shard(jnp.sum(exp_shifted, axis=-1, keepdims=True), None)
    loss = jnp.einsum(f"{spec},{spec}->", jnp.log(sum_exp) - shifted, tgt)
    loss = loss / normalization
    grad = (exp_shifted / sum_exp - tgt) / normalization
    del spec, tgt, shifted, exp_shifted, sum_exp, src

    def grad_fn(g: jnp.ndarray) -> typing.Tuple[jnp.ndarray, None]:
        return g * grad, None

    return loss, grad_fn


def body_ctx(ctx: Context, src: jnp.ndarray) -> jnp.ndarray:
    src = input_embed(ctx, src)
    zero = jnp.zeros_like(src)
    src = (ctx.parameters, src, zero, src, zero)
    for i in range(ctx.model.depth):
        is_last = (i + 1) == ctx.model.depth
        src = reversible(ctx, exec_fn(instance_norm, attention), is_last)(src)
        src = reversible(ctx, exec_fn(instance_norm, group_feed_forward), is_last)(src)
    src = src[1] + src[3]
    src = instance_norm(ctx, src)
    return output_embed(ctx, src)


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return cross_entropy_loss(body_ctx(ctx, src), tgt)


def get_current_lr(ctx: Context, current_step: jnp.ndarray) -> jnp.ndarray:
    opt = ctx.optimizer
    learning_rate = opt.learning_rate
    learning_rate *= jnp.minimum(current_step, opt.warmup_end).astype(jnp.float32) / opt.warmup_end
    learning_rate *= (1 - opt.exponential_decay) ** relu(current_step.astype(jnp.float32) - opt.warmup_end)
    return learning_rate.astype(ctx.model.dtype)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray], current_step: jnp.ndarray):
    ctx = ctx.add_to_prefix("optimizer")
    lr = -get_current_lr(ctx, current_step)
    for param_name, grad in grads.items():
        inner_ctx = ctx.add_to_prefix(param_name)
        if "optimizer" in param_name:
            continue
        grad = adaptive_gradient_clipping(inner_ctx, param_name, grad)
        grad = sm3(inner_ctx, param_name, grad)
        grad = momentum(inner_ctx, param_name, grad)
        ctx.parameters[param_name] = ctx.parameters[param_name] + grad * lr


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(wctx.ctx.parameters, wctx.data[wctx.current_step % wctx.ctx.training.device_steps])
    update(wctx.ctx, grads, wctx.current_step)
    wctx.loss += loss
    wctx.current_step += 1
    return wctx.serialize()


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhileTrainContext(while_ctx_dict)
    return jnp.not_equal(jnp.mod(wctx.current_step, wctx.ctx.training.device_steps + 1), 0)


def jitless_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    return lax.while_loop(cond_fn, train_step, while_ctx_dict)


def sharding(ctx: Context, dims: typing.List[str]):
    out = []
    for d in dims:
        if d == ctx.dims.batch:
            out.append("data_parallel")
        if d == ctx.dims.heads:
            out.append("model_parallel")
        else:
            out.append(None)
    return PartitionSpec(*out)


def timeit(text: str, fn, *args, pad=50):
    start_time = time.time()
    print(f'{text}..', end='', flush=True)
    out = fn(*args)
    print(f"{' ' * (pad - len(text))}Took:{time.time() - start_time:9.2f}s", flush=True)
    return out


def train_loop(wctx: WhileTrainContext, step: typing.Callable):
    ctx = [wctx]

    def _fn(dat: jnp.ndarray) -> WhileTrainContext:
        w = ctx[0](dat)
        w.loss = jnp.zeros_like(w.loss)
        w.current_step = w.current_step + 1
        ctx[0] = WhileTrainContext(step(w.serialize()))
        return ctx[0]

    return _fn


def main():
    wctx = WhileTrainContext()
    ctx = wctx.ctx
    ctx.is_initializing = True
    total_steps = ctx.training.steps * ctx.training.device_steps
    data = timeit("Initializing dataset", text_dataset, ctx)
    inp = timeit("Enqueueing first batch", next, data)[0, 0]
    timeit("Acquiring forward parameters", body_ctx, ctx, inp)
    parameter_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items())
    timeit("Acquiring optimizer parameters", update, ctx,
           {name: jnp.zeros_like(param) for name, param in ctx.parameters.items()},
           jnp.zeros([], dtype=ctx.model.dtype))
    buffer_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items()) - parameter_count

    partition = {'parameters': {name: sharding(ctx, dims) for name, dims in ctx.parameter_dims.items()},
                 'data': PartitionSpec(None, None, "data_parallel", None), 'current_step': None, 'loss': None}
    step = train_loop(wctx, timeit("JITing model", pjit.pjit, jitless_step, (partition,), partition))

    mesh_devices = np.array(jax.devices()).reshape(ctx.training.data_parallel, ctx.training.model_parallel)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        timeit("Compiling model and performing first step", step, next(data))
        print(f"\n\nParameters: {parameter_count:,}\nBuffers:    {buffer_count:,}\n\n")

        start_time = time.time()
        for idx, dat in enumerate(data):
            wctx = step(dat)
            if idx % ctx.training.print_interval == 0:
                print(f'[{idx * ctx.training.device_steps:{len(str(total_steps))}d}/{total_steps}] '
                      f'Loss: {wctx.loss / ctx.training.device_steps:6.3f} - '
                      f'LearningRate: {float(get_current_lr(ctx, wctx.current_step)):.5f} - '
                      f'Took: {time.time() - start_time:10.6f}s')
                start_time = time.time()
            if ctx.training.trace.do_trace:
                if idx == ctx.training.trace.start_step:
                    jax.profiler.start_trace(ctx.training.trace.output_path)
                if idx == ctx.training.trace.stop_step:
                    jax.profiler.stop_trace()


if __name__ == '__main__':
    main()
