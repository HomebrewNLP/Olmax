import jax
from jax import lax, numpy as jnp

from src.backend import matmul
from src.constants import ParallelAxes
from src.context import Context
from src.model.loss import cross_entropy_loss


def random(div: float, *shape: int):
    fn = jax.pmap(lambda x: jax.random.normal(jax.random.PRNGKey(x), shape) / div)
    local_devices = jax.local_device_count()
    seeds = jnp.arange(local_devices * jax.process_index(), local_devices * (1 + jax.process_index()))
    return fn(seeds)


def naive_loss(x, y):
    tmp = lax.psum(matmul(x[0], x[1]), ParallelAxes.model)
    pos = (jax.nn.logsumexp(tmp, -1) / y.size).sum()
    neg = (jnp.take_along_axis(tmp.reshape(-1, tmp.shape[-1]), y.reshape(-1, 1), -1) / y.size).sum()
    return pos - neg


def main():
    ctx = Context()
    key = jax.random.PRNGKey(0)
    k0, k1, k2 = jax.random.split(key, 3)
    tgt = jax.random.randint(k0, (ctx.dims.batch, ctx.dims.sequence), 0, ctx.dims.vocab)

    div = (ctx.dims.features * ctx.dims.heads) ** 0.5
    src = random(div, ctx.dims.batch, ctx.dims.sequence, ctx.dims.features)
    wgt = random(div, ctx.dims.features, ctx.dims.vocab)

    inp = src, wgt
    grad0 = jax.pmap(jax.grad(lambda x: cross_entropy_loss(ctx, x, tgt)[0]), ParallelAxes.model)(inp)
    grad1 = jax.pmap(jax.grad(lambda x: naive_loss(x, tgt)), ParallelAxes.model)(inp)

    _fn = jax.pmap(lambda x: (lax.pmax(x.max(), "i"), lax.pmin(x.min(), "i"), lax.pmean(x.mean(), "i"),
                              lax.pmean(jnp.square(x - x.mean()).mean(), "i")), "i")

    def statistics(name: str, var: jnp.ndarray):
        max, min, mean, std = [float(a[0]) for a in _fn(var)]
        print(f"{name}: {max=}, {min=}, {mean=}, {std=}")

    for g0, g1 in zip(grad0, grad1):
        statistics("Grad0", g0)
        statistics("Grad1", g1)
        statistics("abs(Grad0 - Grad1)", jax.pmap(lambda x, y: jnp.abs(x - y), "i")(g0, g1))
        statistics("abs(Grad0 / Grad1)", jax.pmap(lambda x, y: jnp.abs(x - y), "i")(g0, g1))
        allclose = jax.pmap(lambda x, y: lax.psum(jnp.allclose(x, y).astype(jnp.float32), "i"), "i")(g0, g1)[0]
        print(f'{allclose=}/{jax.device_count()}')
        if allclose < jax.device_count():
            raise ValueError


if __name__ == '__main__':
    main()
