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

from context import Context, WhileContext
from data import text_dataset

warnings.filterwarnings("ignore", message=".*is an experimental feature and probably has bugs!.*")


# jax.config.update("jax_disable_jit", True)

def dims_to_shape(ctx: Context, dims: typing.List[str]) -> typing.List[int]:
    return [ctx.dims.dim_sizes[d] for d in dims]


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


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axis=-1) -> jnp.ndarray:
    n_rows, n_cols = util.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim) * ctx.init_scale
    if n_rows < n_cols:
        out = out.T
    return jnp.moveaxis(jnp.reshape(out, tuple(np.delete(shape, column_axis)) + (shape[column_axis],)), -1, column_axis)


def get_param(ctx: Context, name: str, shape: typing.Optional[typing.List[str]] = None,
              std: typing.Optional[float] = None, mean: typing.Optional[float] = None,
              column_axis: typing.Optional[int] = None) -> jnp.ndarray:
    name = ctx.add_to_prefix(name, count=False).global_prefix
    if name not in ctx.parameters:
        ctx.parameter_dims[name] = shape
        shape = [ctx.dims.dim_sizes[dim] for dim in shape]
        if std is None and mean is None:
            ctx.parameters[name] = orthogonal_init(ctx, shape, -1 if column_axis is None else column_axis)
        else:
            ctx.parameters[name] = random.normal(ctx.prng_key, shape, ctx.dtype)
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
    do_clip = jnp.greater(grd_norm * jnp.reciprocal(wgt_norm), ctx.gradient_clip)
    clipped = wgt_norm * jnp.reciprocal(grd_norm) * ctx.gradient_clip * grad
    return clipped * do_clip + grad * (1 - do_clip)


def momentum(ctx: Context, param_name: str, grad: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("momentum", count=False)
    state = zero_param(ctx, "momentum_buffer", ctx.parameter_dims.get(param_name))
    new_state = ctx.momentum_beta * state + grad
    ctx.parameters[ctx.add_to_prefix("momentum_buffer", count=False).global_prefix] = new_state
    if not ctx.nesterov_momentum:
        return new_state
    return grad + ctx.momentum_beta * new_state


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))


def linear(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("linear")
    spec = base_spec(inp)
    if inp.ndim == 3:
        shape = [ctx.dims.intermediate_feed_forward, ctx.dims.heads, ctx.dims.features_per_head]
        spec = f'{spec},{spec[-1]}yz->{spec[:-1]}yz'
        head_dim = None
        column_axis = 0
    else:
        shape = [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.intermediate_feed_forward]
        spec = f'{spec},{spec[-2:]}z->{spec[:-2]}z'
        head_dim = -2
        column_axis = -1
    return shard(jnp.einsum(spec, inp, get_param(ctx, "weight", shape, column_axis=column_axis)), head_dim)


def relu(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(inp, 0)


def feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("feed_forward")
    return linear(ctx, relu(linear(ctx, inp)))


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def input_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("input_embed")

    spec = base_spec(inp)
    embd = get_param(ctx, "weight", [ctx.dims.vocab, ctx.dims.heads, ctx.dims.features_per_head], ctx.embedding_std)
    out = shard(jnp.einsum(f"{spec}x,xyz->{spec}yz", one_hot(inp, ctx.data.vocab_size), embd))

    position_shape = dims_to_shape(ctx, [ctx.dims.sequence])
    feature_shape = dims_to_shape(ctx, [ctx.dims.heads, ctx.dims.features_per_head])
    position_count = util.prod(position_shape)
    feature_count = util.prod(feature_shape)
    positions = jnp.reshape(jnp.arange(0, position_shape), (-1, 1, 1))
    features = jnp.arange(0, feature_count)
    features = jnp.reshape(features, [1] + feature_shape) * 4 / feature_count
    features = jnp.exp(features - math.log(position_count / 2 / math.pi))
    return out + jnp.sin(features * positions) * ctx.embedding_std


def output_embed(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("output_embed")
    spec = base_spec(inp)[:-2]
    embd = get_param(ctx, "weight", [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.vocab])
    return shard(jnp.einsum(f"{spec}xy,xyz->{spec}z", inp, embd), None)


REVERSIBLE_CTX = typing.Tuple[typing.Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]


def reversible(ctx: Context, fn: typing.Callable):
    @jax.custom_vjp
    def reversible_half_residual(inp: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
        params, x00, x01, x10, x11 = inp
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = ctx.parameters, x10, x11, x00 + fn(new_ctx, x10), x01
        return out

    def reversible_forward(inp: REVERSIBLE_CTX) -> typing.Tuple[REVERSIBLE_CTX, REVERSIBLE_CTX]:
        out = reversible_half_residual(inp)
        return out, out

    def reversible_backward(inp: REVERSIBLE_CTX, dy: REVERSIBLE_CTX) -> REVERSIBLE_CTX:
        params, x10, x11, y00, x01 = inp
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        x00 = y00 - fn(new_ctx, x10)
        _, grad_fn = jax.vjp(reversible_forward, (params, x00, x01, x10, x11))
        d_params, dx00, dx01, dx10, dx11 = grad_fn(dy)
        return d_params, dx00, x00, x01 + dx10, x10

    reversible_half_residual.defvjp(reversible_forward, reversible_backward)
    return reversible_half_residual


def attention(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    base = linear(ctx, inp)
    qry = linear(ctx, base)
    key = linear(ctx, base)
    val = linear(ctx, base)
    spec = base_spec(qry)
    anonymous_spec = spec.replace(spec[-2], "z")
    logit = shard(jnp.einsum(f'{spec},{anonymous_spec}->{spec[:-1]}z', qry / qry.shape[-1] ** 0.5, key))
    if ctx.masked_attention:
        mask = jnp.reshape(jnp.arange(0, qry.shape[-2]), (1, -1)) > jnp.reshape(jnp.arange(0, qry.shape[-2]), (-1, 1))
        logit += mask * -1e30
    logit = jnp.exp(logit - lax.stop_gradient(logit).max(-1, keepdims=True))
    return shard(jnp.einsum(f'{anonymous_spec},{spec[:-1]}z->{spec}', val, logit / logit.sum(-1, keepdims=True)))


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("instance_norm")
    shape = ["one"] * (inp.ndim - 2) + [ctx.dims.heads, ctx.dims.features_per_head]
    inp = inp - shard(inp.mean(-1, keepdims=True), None)
    scale = lax.rsqrt(ctx.norm_eps + shard(jnp.square(inp).sum(-1, keepdims=True), None))
    scale = scale * get_param(ctx, "scale", shape, ctx.norm_std, 1)
    return scale * inp + get_param(ctx, "shift", shape, ctx.norm_std)


def exec_fn(*fns: typing.Callable) -> typing.Callable:
    def _run(ctx: Context, inp: jnp.ndarray):
        for f in fns:
            inp = f(ctx, inp)
        return inp

    return _run


def cross_entropy_loss(ctx: Context, src: jnp.ndarray, tgt: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("cross_entropy_loss")
    spec = base_spec(src)
    max_src = shard(lax.stop_gradient(src).max(-1, keepdims=True), None)
    log_z = jnp.log(shard(jnp.exp(src - max_src).sum(-1, keepdims=True), None)) + max_src
    loss = jnp.einsum(f"{spec},{spec}->", src - log_z, one_hot(tgt, ctx.data.vocab_size))
    return (jnp.square(log_z).sum() * ctx.z_loss - loss) / tgt.size


def compute_ctx(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    src, tgt = inp
    src = input_embed(ctx, src)
    src = (ctx.parameters, src, jnp.zeros_like(src), src, jnp.zeros_like(src))
    for _ in range(ctx.depth):
        src = reversible(ctx, exec_fn(instance_norm, feed_forward))(src)
        src = reversible(ctx, exec_fn(instance_norm, attention))(src)
    src = src[1] + src[3]
    src = instance_norm(ctx, src)
    src = output_embed(ctx, src)
    return cross_entropy_loss(ctx, src, tgt)


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameters = params
    return compute_ctx(ctx, inp)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray]):
    ctx = ctx.add_to_prefix("optimizer")
    for param_name, grad in grads.items():
        inner_ctx = ctx.add_to_prefix(param_name)
        if "optimizer" in param_name:
            continue
        grad = adaptive_gradient_clipping(inner_ctx, param_name, grad)
        grad = sm3(inner_ctx, param_name, grad)
        grad = momentum(inner_ctx, param_name, grad)
        ctx.parameters[param_name] = ctx.parameters[param_name] + grad * ctx.learning_rate


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(wctx.ctx.parameters, wctx.data[wctx.current_step % wctx.ctx.device_steps])
    update(wctx.ctx, grads)
    wctx.loss += loss
    wctx.current_step += 1
    return wctx.serialize()


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhileContext(while_ctx_dict)
    return jnp.not_equal(jnp.mod(wctx.current_step + 1, wctx.ctx.device_steps), 0)


def jitless_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    wctx = WhileContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx = WhileContext(lax.while_loop(cond_fn, train_step, wctx.serialize()))
    return wctx.loss / wctx.ctx.device_steps, wctx.ctx.parameters


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


def timeit(text: str, fn, *args, pad=45):
    start_time = time.time()
    print(f'{text}..{" " * (pad - len(text))}', end='')
    out = fn(*args)
    print(f"Took:{time.time() - start_time:7.2f}s")
    return out


def main():
    ctx = Context()
    ctx.initializing = True
    data = timeit("Initializing dataset", text_dataset, ctx)
    inp = timeit("Enqueueing first batch", next, data)[0]
    timeit("Acquiring forward parameters", compute_ctx, ctx, inp)
    timeit("Acquiring optimizer parameters", update, ctx,
           {name: jnp.zeros_like(param) for name, param in ctx.parameters.items()})

    partition = {name: sharding(ctx, dims) for name, dims in ctx.parameter_dims.items()}
    step = timeit("JITing model", pjit.pjit, jitless_step,
                  (partition, PartitionSpec(None, None, "data_parallel", None)), (None, partition))

    mesh_devices = np.array(jax.devices()).reshape(ctx.data_parallel, ctx.model_parallel)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        loss, parameters = timeit("Compiling model and performing first step", step, ctx.parameters, next(data))

        print(f"Parameters: {sum(util.prod(param.shape) for name, param in parameters.items())}")

        start_time = time.time()

        for idx, dat in enumerate(data):
            loss, parameters = step(parameters, dat)
            if idx % ctx.print_interval == 0:
                print(f'[{idx * ctx.device_steps:{len(str(ctx.steps * ctx.device_steps))}d}/'
                      f'{ctx.steps * ctx.device_steps}] Loss: {loss:6.3f} - Took: {time.time() - start_time:9.6f}s')
                start_time = time.time()


if __name__ == '__main__':
    main()
