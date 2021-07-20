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


def dims_to_shape(ctx: Context, dims: typing.List[str]):
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


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axis=-1, ) -> jnp.ndarray:
    n_rows, n_cols = util.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim) * ctx.init_scale
    if n_rows < n_cols:
        out = out.T
    return jnp.moveaxis(jnp.reshape(out, tuple(np.delete(shape, column_axis)) + (shape[column_axis],)), -1, column_axis)


def get_param(ctx: Context, name: str, shape: typing.Optional[typing.List[str]] = None) -> jnp.ndarray:
    name = ctx.add_to_prefix(name).global_prefix
    if name not in ctx.parameters:
        ctx.parameter_dims[name] = shape
        ctx.parameters[name] = orthogonal_init(ctx, [ctx.dims.dim_sizes[dim] for dim in shape])
    return ctx.parameters[name]


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))


def linear(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("linear")
    spec = base_spec(inp)
    if inp.ndim == 3:
        shape = [ctx.dims.intermediate_feed_forward, ctx.dims.heads, ctx.dims.features_per_head]
        spec = f'{spec},{spec[-1]}yz->{spec[:-1]}yz'
        head_dim = None
    else:
        shape = [ctx.dims.heads, ctx.dims.features_per_head, ctx.dims.intermediate_feed_forward]
        spec = f'{spec},{spec[-2:]}z->{spec[:-2]}z'
        head_dim = -2
    return shard(jnp.einsum(spec, inp, get_param(ctx, "weight", shape)), head_dim)


def relu(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(inp, 0)


def feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("feed_forward")
    return linear(ctx, relu(linear(ctx, inp)))


def attention(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    base = linear(ctx, inp)
    qry = linear(ctx, base)
    key = linear(ctx, base)
    val = linear(ctx, base)
    spec = base_spec(qry)
    anonymous_spec = spec.replace(spec[-2], "z")
    logit = shard(jnp.einsum(f'{spec},{anonymous_spec}->{spec[:-1]}z', qry, key) / qry.shape[-1])
    if ctx.masked_attention:
        mask = jnp.reshape(jnp.arange(0, qry.shape[-2]), (1, -1)) > jnp.reshape(jnp.arange(0, qry.shape[-2]), (-1, 1))
        logit += mask * -1e30
    logit = jnp.exp(logit - lax.stop_gradient(logit.max(-1, keepdims=True)))
    logit /= logit.sum(-1, keepdims=True)
    return shard(jnp.einsum(f'{anonymous_spec},{spec[:-1]}z->{spec}', val, logit))


def instance_norm(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("instance_norm")
    shape = ["one"] * (inp.ndim - 2) + [ctx.dims.heads, ctx.dims.features_per_head]
    inp = inp - shard(inp.mean(-1, keepdims=True), None)
    inp = inp * (lax.rsqrt(shard(jnp.square(inp).sum(-1, keepdims=True), None)) * get_param(ctx, "scale", shape))
    return inp + get_param(ctx, "shift", shape)


def compute_ctx(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    src, tgt = inp
    for _ in range(ctx.depth):
        src += feed_forward(ctx, instance_norm(ctx, src))
        src += attention(ctx, instance_norm(ctx, src))
    return jnp.square(src - tgt).mean()


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameters = params
    return compute_ctx(ctx, inp)


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray]) -> typing.Dict[str, jnp.ndarray]:
    return {k: p + grads[k] * ctx.learning_rate for k, p in ctx.parameters.items()}


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(wctx.ctx.parameters, wctx.data[wctx.current_step % wctx.ctx.device_steps])
    wctx.ctx.parameters = update(wctx.ctx, grads)
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


def main():
    ctx = Context()
    ctx.initializing = True
    data = text_dataset(ctx)
    print("Acquiring parameters and graph..        ", end='', flush=True)
    start_time = time.time()
    compute_ctx(ctx, next(data)[0])
    print(f"Took {time.time() - start_time:.1f}s")

    parameters = ctx.parameters

    print("Compiling model..                       ", end='', flush=True)

    partition = {name: sharding(ctx, dims) for name, dims in ctx.parameter_dims.items()}
    step = pjit.pjit(jitless_step,
                     in_axis_resources=(partition, PartitionSpec(None, None, "data_parallel", None, None, None)),
                     out_axis_resources=(None, partition))
    mesh_devices = np.array(jax.devices()).reshape(ctx.data_parallel, ctx.model_parallel)
    with mesh(mesh_devices, ('data_parallel', 'model_parallel')):
        start_time = time.time()
        step(parameters, next(data))
        print(f"Took {time.time() - start_time:.1f}s")

        print(f"Parameters: {sum(util.prod(param.shape) for name, param in parameters.items())}")

        start_time = time.time()

        for idx, dat in enumerate(data):
            loss, parameters = step(parameters, dat)
            if idx % ctx.print_interval == 0:
                print(
                    f'[{idx * ctx.device_steps:{len(str(ctx.steps * ctx.device_steps))}d}/{ctx.steps * ctx.device_steps}]'
                    f' Loss: {loss:6.3f} - Took: {time.time() - start_time:9.6f}s')
                start_time = time.time()


if __name__ == '__main__':
    main()
