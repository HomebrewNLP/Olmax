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
        self.device_steps = 200
        self.steps = 10
        self.features_per_head = 16
        self.head_count = 1
        self.group_linear_factor = 2
        self.batch_size = 7
        self.base = self.features_per_head * self.head_count
        self.out = self.base * self.group_linear_factor
        self.dtype = jnp.float32
        self.init_scale = 1.0


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


def compute(params: typing.Dict[str, jnp.ndarray]) -> jnp.ndarray:
    ctx = Context()
    ctx.parameter_dict = params
    return linear(relu(linear(input_embedding(ctx, "embd"), ctx, "lin0")), ctx, "lin1").mean()


def update(ctx: Context, grads: typing.List[jnp.ndarray]) -> typing.Dict[str, jnp.ndarray]:
    return {k: p - g * ctx.learning_rate for (k, p), g in zip(ctx.parameter_dict.items(), grads.values())}


def train_step(ctx: Context) -> typing.Tuple[jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(ctx.parameter_dict)
    return loss, update(ctx, grads)


@jax.jit
def step(parameter_dict: typing.Dict[str,jnp.ndarray]) -> typing.Tuple[jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    ctx = Context()
    ctx.parameter_dict = parameter_dict
    loss = jnp.zeros([])
    for _ in range(ctx.device_steps):
        out, ctx.parameter_dict = train_step(ctx)
        loss += out
    return loss / ctx.device_steps, ctx.parameter_dict


def main():
    ctx = Context()
    compute(ctx.parameter_dict)
    parameter_dict = ctx.parameter_dict
    for _ in range(ctx.steps):
        start_time = time.time()
        loss, parameter_dict = step(parameter_dict)
        print(loss, time.time() - start_time)


if __name__ == '__main__':
    main()
