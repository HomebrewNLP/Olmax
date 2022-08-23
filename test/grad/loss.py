import jax
from jax import lax, numpy as jnp

from src.backend import matmul
from src.constants import ParallelAxes
from src.context import Context
from src.model.loss import cross_entropy_loss


def random(*shape: int):
    fn = jax.pmap(lambda x: jax.random.normal(jax.random.PRNGKey(x), shape))
    local_devices = jax.local_device_count()
    seeds = jnp.arange(local_devices * jax.process_index(), local_devices * (1 + jax.process_index()))
    return fn(seeds)


def naive_loss(x, y):
    tmp = lax.psum(matmul(x[0], x[1]), ParallelAxes.model)
    pos = jax.nn.logsumexp(tmp, -1).mean()
    neg = jnp.take_along_axis(tmp.reshape(-1, tmp.shape[-1]), y.reshape(-1, 1), -1).mean()
    return pos - neg


def main():
    ctx = Context()
    key = jax.random.PRNGKey(0)
    k0, k1, k2 = jax.random.split(key, 3)
    tgt = jax.random.randint(k0, (ctx.dims.batch, ctx.dims.sequence), 0, ctx.dims.vocab)

    src = random(ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
    wgt = random(ctx.dims.vocab, ctx.dims.features) / ctx.dims.features

    inp = src, wgt
    grad0 = jax.pmap(jax.grad(lambda x: cross_entropy_loss(ctx, x, tgt)[0]), ParallelAxes.model)(inp)
    grad1 = jax.pmap(jax.grad(lambda x: naive_loss(x, tgt)), ParallelAxes.model)(inp)

    for g0, g1 in zip(grad0, grad1):
        max_abs_dist = jax.pmap(lambda x, y: lax.pmax(jnp.abs(x - y).max(), "i"), "i")(g0, g1)[0]
        max_rel_dist = jax.pmap(lambda x, y: lax.pmax(jnp.abs(x / y).max(), "i"), "i")(g0, g1)[0]
        print(max_abs_dist, max_rel_dist)
        allclose = jax.pmap(lambda x, y: lax.psum(jnp.allclose(x, y).astype(jnp.float32), "i"), "i")(g0, g1)[0]
        if not allclose:
            raise ValueError


if __name__ == '__main__':
    main()
