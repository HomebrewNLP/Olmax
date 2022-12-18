import jax
from jax import numpy as jnp


def activate_forward(inp: jax.Array) -> jax.Array:
    return inp * activate_grad(inp)


def activate_grad(inp: jax.Array) -> jax.Array:
    return jnp.where(inp < 0, 0.01, 1)


def activate(inp: jax.Array) -> jax.Array:
    @jax.custom_gradient
    def _fn(x: jax.Array):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
