import copy
import math
import time
import typing
import warnings

import jax
import jax._src.util as util
import numpy as np
import yaml
from jax import lax, numpy as jnp
from jax.experimental import PartitionSpec
from jax.experimental import pjit
from jax.experimental.maps import mesh

from backend import default, get_param, shard, dims_to_shape
from context import Context, WhileTrainContext
from data import text_dataset
from optimizer import get_current_lr, update

REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


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


def activate(ctx, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(src: jnp.ndarray):
        out = jax.nn.leaky_relu(src, ctx.model.leaky_relu_slope)

        def _grad_fn(dy: jnp.ndarray):
            return activation_backward(ctx, dy, out)

        return out, _grad_fn

    return _fn(inp)


def activation_backward(ctx: Context, dy: jnp.ndarray, inp: jnp.ndarray) -> jnp.ndarray:
    scale = (1 - ctx.model.leaky_relu_slope) * jnp.greater(inp, 0).astype(ctx.model.dtype) + ctx.model.leaky_relu_slope
    return dy * scale


def instance_norm_forward(ctx: Context, inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    mean = shard(inp.mean(-1, keepdims=True))
    out = inp - mean
    scale = lax.rsqrt(ctx.model.norm_eps + shard(jnp.square(out).mean(-1, keepdims=True)))
    return out * scale, mean, scale


def instance_norm_backward(dy: jnp.ndarray, inp: jnp.ndarray, out: jnp.ndarray, scale: jnp.ndarray) -> jnp.ndarray:
    dy *= scale / inp.shape[-1]
    dx1 = dy + out * shard(jnp.sum(dy * out, -1, keepdims=True)) / (- inp.shape[-1] ** 2)
    return dx1 - shard(jnp.sum(dx1, -1, keepdims=True))


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    @jax.custom_gradient
    def _fn(src: jnp.ndarray):
        out, mean, scale = instance_norm_forward(ctx, src)

        def _grad(dy: jnp.ndarray) -> jnp.ndarray:
            return instance_norm_backward(dy, out / scale + mean, out, scale)

        return out, _grad

    return _fn(inp)


def group_feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("group_feed_forward")
    ndim = inp.ndim
    features = [ctx.dims.features_per_head, ctx.dims.intermediate_feed_forward]
    inp_weight = get_param(ctx, "inp_weight", [ctx.dims.heads] + features, scale=1 / ctx.model.activation_std)
    out_weight = get_param(ctx, "out_weight", [ctx.dims.heads] + features[::-1], scale=ctx.model.depth ** -0.5)
    if ctx.is_initializing:
        return inp

    normed = instance_norm(ctx, inp)
    mid = activate(ctx, shard(dot_general(normed, inp_weight, (ndim - 1,), (1,), (ndim - 2,), (0,)), 0, 1))
    out = shard(dot_general(activate(ctx, mid), out_weight, (ndim - 1,), (1,), (0,), (0,)), 0, 1)
    out = shard(out.transpose(tuple(range(1, ndim - 1)) + (0, ndim - 1)))
    return out


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")

    inp_embd = get_param(ctx, "inp_embd", [ctx.dims.vocab, ctx.dims.intermediate_attention],
                         1 / ctx.model.activation_std)
    out_embd = get_param(ctx, "out_embd",
                         [ctx.dims.intermediate_attention, ctx.dims.heads, ctx.dims.features_per_head],
                         column_axes=(1, 2))
    if ctx.is_initializing:
        return jnp.zeros([1] * (inp.ndim + 1))
    batch_dims = range(inp.ndim)
    ndim = inp.ndim + 2

    @jax.custom_gradient
    def _fn(src: jnp.ndarray, i_e: jnp.ndarray, o_e: jnp.ndarray):
        mid = activate(ctx, shard(dot_product(one_hot(src, ctx.data.vocab_size).astype(ctx.model.dtype),
                                              i_e, -1, 0), None))

        def _grad_fn(dy: jnp.ndarray) -> typing.Tuple[None, jnp.ndarray, jnp.ndarray]:
            one_hot_src = one_hot(src, ctx.data.vocab_size).astype(ctx.model.dtype)
            o_e_grad = dot_general(mid, dy, batch_dims, batch_dims)
            mid_grad = activation_backward(ctx, dot_general(dy, o_e, (ndim - 2, ndim - 1), (1, 2)), mid)
            i_e_grad = dot_general(one_hot_src, mid_grad, batch_dims, batch_dims)
            return None, i_e_grad, o_e_grad

        out = shard(dot_product(mid, o_e, -1, 0))
        position_shape = dims_to_shape(ctx, [ctx.dims.sequence])
        feature_shape = dims_to_shape(ctx, [ctx.dims.heads, ctx.dims.features_per_head])
        position_count = util.prod(position_shape)
        feature_count = util.prod(feature_shape)
        positions = jnp.reshape(jnp.arange(0, position_shape), (-1, 1, 1))
        features = jnp.arange(0, feature_count)
        features = shard(jnp.reshape(features, [1] + feature_shape) * 4 / feature_count, 1, None)
        features = jnp.exp(shard(features - math.log(position_count / 2 / math.pi), 1))
        pos_embd = jnp.sin(features * positions)

        return out + pos_embd.astype(out.dtype), _grad_fn

    return _fn(inp, inp_embd, out_embd)


def output_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    embd = get_param(ctx, "weight", [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.vocab], 0, 0)
    if ctx.is_initializing:
        return inp
    return shard(dot_product(inp, embd, -2, 0, -1, 1), None)


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


def attention(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    feature_dims = [ctx.dims.heads, ctx.dims.features_per_head]
    base_param = get_param(ctx, "base", feature_dims + [ctx.dims.intermediate_attention],
                           scale=1 / ctx.model.activation_std)
    key_param = get_param(ctx, "key", [ctx.dims.intermediate_attention] + feature_dims, column_axes=(1, 2))
    qry_param = get_param(ctx, "qry", [ctx.dims.intermediate_attention] + feature_dims, column_axes=(1, 2))
    val_param = get_param(ctx, "val", [ctx.dims.intermediate_attention] + feature_dims, column_axes=(1, 2),
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
    base = activate(ctx, shard(dot_product(base, base_param, -2, 0, -1, 1), None))
    key = shard(dot_product(base, key_param, -1, 0))
    qry = shard(dot_product(base, qry_param, -1, 0))
    val = shard(dot_product(base, val_param, -1, 0))

    key = shard(key.transpose(key_permute), -3) * inp.shape[-1] ** -0.5
    val = shard(val.transpose(key_permute), -3)
    qry = shard(qry.transpose(qry_permute), -3)
    lgt = shard(dot_general(key, qry, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)
    lgt = softmax(ctx, lgt)

    out = shard(dot_general(lgt, val, (feature_dim,), (head_dim,), batch_seq, batch_seq), -3)
    return shard(out.transpose(key_permute))


def cross_entropy_loss(src: jnp.ndarray, tgt: jnp.ndarray):
    tgt = shard(one_hot(tgt.astype(src.dtype), src.shape[-1]), None)
    shifted = src - shard(src.max(axis=-1, keepdims=True), None)
    exp_shifted = jnp.exp(shifted)
    sum_exp = shard(jnp.sum(exp_shifted, axis=-1, keepdims=True), None)
    return shard(((jnp.log(sum_exp) - shifted) * tgt).sum(tuple(range(1, tgt.ndim))), None)


def body_ctx(ctx: Context, src: jnp.ndarray) -> jnp.ndarray:
    src = input_embed(ctx, src)
    zero = shard(jnp.zeros_like(src))
    src = (ctx.parameters, src, zero, src, zero)
    for i in range(ctx.model.depth):
        src = reversible(ctx, attention, (i + 1) == ctx.model.depth)(src)
        src = reversible(ctx, group_feed_forward, (i + 1) == ctx.model.depth)(src)
    return output_embed(ctx, src[1] + src[3])


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, jnp.ndarray]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    unreduced_loss = cross_entropy_loss(body_ctx(ctx, shard(src, None)), shard(tgt, None))
    top_loss = loss = unreduced_loss.sum() / tgt.size
    top_k = math.ceil(ctx.dims.sizes.batch * ctx.training.loss_top_p / ctx.training.loss_top_snap)
    top_k *= ctx.training.loss_top_snap
    if ctx.training.loss_top_p < 1 and top_k < ctx.dims.sizes.batch:
        top_loss, _ = lax.top_k(unreduced_loss, top_k)
        top_loss = top_loss.sum() / top_k
    return top_loss, loss


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileTrainContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0, True)
    (top_loss, loss), grads = grad_fn(wctx.ctx.parameters,
                                      wctx.data[wctx.current_step % wctx.ctx.training.device_steps])
    update(wctx.ctx, grads, wctx.current_step)
    wctx.loss += loss
    wctx.top_loss += top_loss
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
        w.top_loss = jnp.zeros_like(w.loss)
        w.current_step = w.current_step + 1
        ctx[0] = WhileTrainContext(step(w.serialize()))
        return ctx[0]

    return _fn


def main():
    warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")
    # jax.config.update("jax_disable_jit", True)
    wctx = WhileTrainContext()
    ctx = wctx.ctx
    print(yaml.dump(ctx.config(), indent=4))
    ctx.is_initializing = True
    total_steps = ctx.training.steps * ctx.training.device_steps
    data = timeit("Initializing dataset", text_dataset, ctx)
    inp = timeit("Enqueueing first batch", next, data)[0, 0]
    timeit("Acquiring forward parameters", body_ctx, ctx, inp)
    parameter_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items())
    timeit("Acquiring optimizer parameters", update, ctx,
           {name: jnp.zeros_like(param) for name, param in ctx.parameters.items()},
           jnp.ones([], dtype=ctx.model.dtype))
    buffer_count = sum(util.prod(param.shape) for name, param in ctx.parameters.items()) - parameter_count

    partition = {'parameters': {name: sharding(ctx, dims) for name, dims in ctx.parameter_dims.items()},
                 'data': PartitionSpec(None, None, "data_parallel", None), 'current_step': None, 'loss': None,
                 'top_loss': None}
    step = train_loop(wctx, timeit("JITing model", pjit.pjit, jitless_step, (partition,), partition))

    mesh_devices = np.array(jax.devices()).reshape(ctx.training.data_parallel, ctx.training.model_parallel)
    global_start = time.time()
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        timeit("Compiling model and performing first step", step, next(data))
        print(f"\n\nParameters: {parameter_count:,}\nBuffers:    {buffer_count:,}\n\n")

        start_time = time.time()
        for idx, dat in enumerate(data):
            wctx = step(dat)
            if idx % ctx.training.print_interval == 0:
                millions_processed = ctx.training.device_steps * ctx.dims.sizes.sequence * ctx.dims.sizes.batch
                print(f'[{idx * ctx.training.device_steps:{len(str(total_steps))}d}/{total_steps}] '
                      f'Loss: {wctx.loss / ctx.training.device_steps:6.3f} - '
                      f'TopLoss: {wctx.top_loss / ctx.training.device_steps:8.3f} | '
                      f'LearningRate: {float(get_current_lr(ctx, wctx.current_step)):.5f} | '
                      f'StepTime: {time.time() - start_time:10.6f}s - '
                      f'Rate: {millions_processed * (idx + 1) / (time.time() - global_start):9,.1f} Tokens/s')
                start_time = time.time()
            if ctx.training.trace.do_trace:
                if idx == ctx.training.trace.start_step:
                    jax.profiler.start_trace(ctx.training.trace.output_path)
                if idx == ctx.training.trace.stop_step:
                    jax.profiler.stop_trace()


if __name__ == '__main__':
    main()
