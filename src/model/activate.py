import jax
from jax import lax, numpy as jnp

from src.context import Context


def softplus(inp: jnp.ndarray):
    return lax.max(inp, 0) + lax.log1p(lax.exp(-lax.abs(inp)))


def activate_forward(inp: jnp.ndarray) -> jnp.ndarray:
    return inp * lax.tanh(softplus(inp))


def activate_grad(inp: jnp.ndarray) -> jnp.ndarray:
    return 1 - lax.square(lax.tanh(softplus(inp))) * inp * jax.nn.sigmoid(inp)


def activate(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(x: jnp.ndarray):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
