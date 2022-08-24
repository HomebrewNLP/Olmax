import jax
from jax import numpy as jnp

from src.context import Context


def activate_forward(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(inp > 0, inp, inp * 0.01)


def activate_grad(inp: jnp.ndarray) -> jnp.ndarray:
    return jnp.where(inp > 0, 1, 0.01)


def activate(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    if ctx.is_initializing:
        return inp

    @jax.custom_gradient
    def _fn(x: jnp.ndarray):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
