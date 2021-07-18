import typing

import jax
import jax._src.util as util
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import time

class Context:
    def __init__(self):
        self.seed = 0
        self.prng_key = random.PRNGKey(self.seed)
        self.learning_rate = 1e-3
        self.parameters: typing.List[jnp.ndarray] = []
        self.device_steps = 200
        self.steps = 10


def orthogonal_init(ctx: Context, shape: typing.List[int], scale=1.0, column_axis=-1, dtype=jnp.float32):
    n_rows, n_cols = util.prod(shape) // shape[column_axis], shape[column_axis]
    matrix_shape = (n_rows, n_cols) if n_rows > n_cols else (n_cols, n_rows)
    out, r = jnp.linalg.qr(random.normal(ctx.prng_key, matrix_shape, dtype))
    out *= lax.broadcast_to_rank(jnp.sign(jnp.diag(r)), rank=out.ndim) * scale
    if n_rows < n_cols:
        out = out.T
    return jnp.moveaxis(jnp.reshape(out, tuple(np.delete(shape, column_axis)) + (shape[column_axis],)), -1, column_axis)


def compute(params: typing.List[jnp.ndarray]) -> jnp.ndarray:
    a, b = params
    c = jnp.einsum("ij,jk->ik", a, b)
    return c.mean()


def update(ctx: Context, grads: typing.List[jnp.ndarray]):
    return [p - g * ctx.learning_rate for p, g in zip(ctx.parameters, grads)]


def train_step(ctx: Context):
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(ctx.parameters)
    return loss, update(ctx, grads)


@jax.jit
def step(parameters: typing.List[jnp.ndarray]) -> typing.Tuple[jnp.ndarray, typing.List[jnp.ndarray]]:
    ctx = Context()
    ctx.parameters = parameters
    loss = jnp.zeros([])
    for _ in range(ctx.device_steps):
        out, ctx.parameters = train_step(ctx)
        loss += out
    return loss / ctx.device_steps, ctx.parameters


def main():
    ctx = Context()
    a = orthogonal_init(ctx, [3, 5])
    b = orthogonal_init(ctx, [5, 2])

    parameters = [a, b]
    for _ in range(ctx.steps):
        start_time = time.time()
        loss, parameters = step(parameters)
        print(loss, time.time() - start_time)


if __name__ == '__main__':
    main()
