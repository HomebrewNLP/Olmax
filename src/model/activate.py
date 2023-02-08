import jax
from jax import lax, numpy as jnp


# [Rethinking Channel Dimensions for Efficient Model Design](https://arxiv.org/abs/2007.00992) showed that
# "complicated" activation functions such as SiLU, GeLU and Mish help increase the rank of the embedding matrix.
# To ensure low-redundancy features, we use a function from that family.
def _tanh_sp(inp: jax.Array) -> jax.Array:  # jax.nn.softplus without nan check
    return lax.tanh(jnp.maximum(inp, 0.) + lax.log1p(lax.exp(-lax.abs(inp))))


def activate_forward(inp: jax.Array) -> jax.Array:
    return inp * _tanh_sp(inp)


def activate_grad(inp: jax.Array) -> jax.Array:
    tanh_sp = _tanh_sp(inp)
    return tanh_sp + inp * jax.nn.sigmoid(inp) * (1 - lax.square(tanh_sp))


def activate(inp: jax.Array) -> jax.Array:
    @jax.custom_gradient
    def _fn(x: jax.Array):
        return activate_forward(x), lambda dy: dy * activate_grad(x)

    return _fn(inp)
