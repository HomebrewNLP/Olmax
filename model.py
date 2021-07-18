import time
import typing

import jax
import jax._src.util as util
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import numpy as np


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


def dataset(ctx: Context):
    shape = [ctx.device_steps, ctx.batch_size, ctx.base]
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
    if name not in ctx.parameter_dict:
        ctx.parameter_dict[name] = orthogonal_init(ctx, shape)
    return ctx.parameter_dict[name]


def linear(inp: jnp.ndarray, params: Context, name: str) -> jnp.ndarray:
    shape = [params.out, params.base]
    if inp.shape[-1] == params.base:
        shape = shape[::-1]
    spec = ''.join(chr(ord('a') + i) for i in range(inp.ndim))
    return jnp.einsum(f'{spec},{spec[-1]}z->{spec[:-1]}z', inp, get_or_create_parameter(params, name, shape))


def input_embedding(params: Context, name: str) -> jnp.ndarray:
    return get_or_create_parameter(params, name, [params.batch_size, params.base])


def relu(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.maximum(inp, 0)


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameter_dict = params
    return linear(relu(linear(inp, ctx, "lin0")), ctx, "lin1").mean()


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
    for dat in data:
        start_time = time.time()
        loss, parameter_dict = step(parameter_dict, dat)
        print(loss, time.time() - start_time)


if __name__ == '__main__':
    main()
