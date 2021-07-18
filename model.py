import time
import typing

import jax
import jax._src.util as util
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import copy


class Context:
    def __init__(self):
        self.seed = 0
        self.prng_key = random.PRNGKey(self.seed)
        self.learning_rate = 1e-3
        self.parameters: typing.List[jnp.ndarray] = []
        self.parameter_dict: typing.Dict[str:jnp.ndarray] = {}
        self.device_steps = 25
        self.steps = 100
        self.features_per_head = 16
        self.head_count = 1
        self.group_linear_factor = 2
        self.batch_size = 7
        self.base = self.features_per_head * self.head_count
        self.out = self.base * self.group_linear_factor
        self.dtype = jnp.float32
        self.init_scale = 1.0
        self.global_prefix = ''
        self.sequence_length = 17
        self.name_cache: typing.Dict[str, int] = {}

    def add_to_prefix(self, appended=""):
        new = copy.copy(self)
        new.global_prefix = self.global_prefix + '/' + self.incremental_name(appended)
        return new

    def incremental_name(self, name):
        if name not in self.name_cache:
            self.name_cache[name] = -1
        self.name_cache[name] += 1
        return f'{name}:{self.name_cache[name]:d}'


def dataset(ctx: Context):
    shape = [ctx.device_steps, ctx.batch_size, ctx.sequence_length, ctx.base]
    size = util.prod(shape)
    for i in range(ctx.steps):
        yield jnp.reshape(jnp.arange(0, size), shape) / size


def orthogonal_init(ctx: Context, shape: typing.List[int], column_axis=-1, ) -> jnp.ndarray:
    n_rows, n_cols = util.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, ctx.dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim) * ctx.init_scale
    if n_rows < n_cols:
        out = out.T
    return jnp.moveaxis(jnp.reshape(out, tuple(np.delete(shape, column_axis)) + (shape[column_axis],)), -1, column_axis)


def get_or_create_parameter(ctx: Context, name: str, shape: typing.Optional[typing.List[int]] = None) -> jnp.ndarray:
    name = ctx.add_to_prefix(name).global_prefix
    if name not in ctx.parameter_dict:
        ctx.parameter_dict[name] = orthogonal_init(ctx, shape)
    return ctx.parameter_dict[name]


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))


def linear(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("linear")
    shape = [ctx.out, ctx.base]
    if inp.shape[-1] == ctx.base:
        shape = shape[::-1]
    spec = base_spec(inp)
    return jnp.einsum(f'{spec},{spec[-1]}z->{spec[:-1]}z', inp, get_or_create_parameter(ctx, "weight", shape))


def input_embedding(ctx: Context, name: str) -> jnp.ndarray:
    return get_or_create_parameter(ctx, name, [ctx.batch_size, ctx.base])


def relu(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(inp, 0)


def feed_forward(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("feed_forward")
    return linear(ctx, relu(linear(ctx, inp)))


def attention(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("attention")
    qry = linear(ctx, inp)
    key = linear(ctx, inp)
    val = linear(ctx, inp)
    spec = base_spec(qry)
    anonymous_spec = spec.replace(spec[-2], "z")
    logit = jnp.einsum(f'{spec},{anonymous_spec}->{spec[:-1]}z', qry, key)
    logit -= jnp.reshape(jnp.arange(0, qry.shape[-2]), (1, -1)) > jnp.reshape(jnp.arange(0, qry.shape[-2]), (-1, 1))
    logit = jnp.exp(logit - lax.stop_gradient(logit.max(-1, keepdims=True)))
    logit /= logit.sum(-1, keepdims=True)
    return linear(ctx, jnp.einsum(f'{anonymous_spec},{spec[:-1]}z->{spec}', val, logit))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameter_dict = params
    return attention(ctx, feed_forward(ctx, inp)).mean()


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray]) -> typing.Dict[str, jnp.ndarray]:
    return {k: p - g * ctx.learning_rate for (k, p), g in zip(ctx.parameter_dict.items(), grads.values())}


def train_step(ctx: Context, inp: jnp.ndarray) -> typing.Tuple[jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(ctx.parameter_dict, inp)
    return loss, update(ctx, grads)


@jax.jit
def step(parameter_dict: typing.Dict[str, jnp.ndarray], data: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    ctx = Context()
    ctx.parameter_dict = parameter_dict
    loss = jnp.zeros([])
    for i in range(ctx.device_steps):
        out, ctx.parameter_dict = train_step(ctx, data[i])
        loss += out
    return loss / ctx.device_steps, ctx.parameter_dict


def main():
    ctx = Context()
    ctx.initializing = True
    data = dataset(ctx)
    compute(ctx.parameter_dict, next(data)[0])
    parameter_dict = ctx.parameter_dict
    for name, param in parameter_dict.items():
        print(name, util.prod(param.shape), param.shape)
    for dat in data:
        start_time = time.time()
        loss, parameter_dict = step(parameter_dict, dat)
        print(loss, time.time() - start_time)


if __name__ == '__main__':
    main()
