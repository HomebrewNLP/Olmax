import copy
import time
import typing

import jax
import jax._src.util as util
import jax.lax as lax
import jax.numpy as jnp
import jax.random as random
import numpy as np


class Context:
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.seed = 0
        self.prng_key = random.PRNGKey(self.seed)
        self.learning_rate = 1e-3
        self.parameters: typing.Dict[str:jnp.ndarray] = {}
        self.device_steps = 1024
        self.steps = 2 ** 16
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
        self.masked_attention = False
        self.print_interval = 16

        if config is not None:
            self.__dict__.update(config)

    def add_to_prefix(self, appended=""):
        new = copy.copy(self)
        new.global_prefix = self.global_prefix + '/' + self.incremental_name(appended)
        return new

    def incremental_name(self, name):
        if name not in self.name_cache:
            self.name_cache[name] = -1
        self.name_cache[name] += 1
        return f'{name}:{self.name_cache[name]:d}'

    def serialize(self):
        return copy.copy(self.__dict__)


class WhileContext:
    def __init__(self, config: typing.Optional[typing.Dict[str, typing.Any]] = None):
        self.ctx = Context()
        self.current_step = jnp.zeros([], dtype=jnp.int64)
        self.data: typing.Optional[jnp.ndarray] = None
        self.loss = jnp.zeros([])

        if config is not None:
            self.ctx.parameters = config['parameters']
            self.loss = config['loss']
            self.current_step = config['current_step']
            self.data = config['data']

    def serialize(self):
        return {'parameters': self.ctx.parameters, 'current_step': self.current_step, 'loss': self.loss,
                'data': self.data}


def dataset(ctx: Context):
    shape = [ctx.device_steps, 2, ctx.batch_size, ctx.sequence_length, ctx.base]
    size = util.prod(shape)
    for i in range(ctx.steps):
        yield jnp.reshape(jnp.cos(jnp.arange(0, size)), shape)


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
    if name not in ctx.parameters:
        ctx.parameters[name] = orthogonal_init(ctx, shape)
    return ctx.parameters[name]


def base_spec(inp: jnp.ndarray) -> str:
    return ''.join(chr(ord('a') + i) for i in range(inp.ndim))


def linear(ctx: Context, inp: jnp.ndarray) -> jnp.ndarray:
    ctx = ctx.add_to_prefix("linear")
    shape = [ctx.out, ctx.base]
    if inp.shape[-1] == ctx.base:
        shape = shape[::-1]
    spec = base_spec(inp)
    return jnp.einsum(f'{spec},{spec[-1]}z->{spec[:-1]}z', inp, get_or_create_parameter(ctx, "weight", shape))


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
    if ctx.masked_attention:
        logit -= jnp.reshape(jnp.arange(0, qry.shape[-2]), (1, -1)) > jnp.reshape(jnp.arange(0, qry.shape[-2]), (-1, 1))
    logit = jnp.exp(logit - lax.stop_gradient(logit.max(-1, keepdims=True)))
    logit /= logit.sum(-1, keepdims=True)
    return linear(ctx, jnp.einsum(f'{anonymous_spec},{spec[:-1]}z->{spec}', val, logit))


def compute(params: typing.Dict[str, jnp.ndarray], inp: jnp.ndarray) -> jnp.ndarray:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return jnp.square(attention(ctx, feed_forward(ctx, src)) - tgt).mean()


def update(ctx: Context, grads: typing.Dict[str, jnp.ndarray]) -> typing.Dict[str, jnp.ndarray]:
    return {k: p - g * ctx.learning_rate for (k, p), g in zip(ctx.parameters.items(), grads.values())}


def train_step(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhileContext(while_ctx_dict)
    grad_fn = jax.value_and_grad(compute, 0)
    loss, grads = grad_fn(wctx.ctx.parameters, wctx.data[wctx.current_step % wctx.ctx.device_steps])
    wctx.ctx.parameters = update(wctx.ctx, grads)
    wctx.loss += loss
    wctx.current_step += 1
    return wctx.serialize()


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhileContext(while_ctx_dict)
    return jnp.not_equal(jnp.mod(wctx.current_step + 1, wctx.ctx.device_steps), 0)


@jax.jit
def step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    wctx = WhileContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx = WhileContext(lax.while_loop(cond_fn, train_step, wctx.serialize()))
    return wctx.loss / wctx.ctx.device_steps, wctx.ctx.parameters


def main():
    ctx = Context()
    ctx.initializing = True
    data = dataset(ctx)
    print("Acquiring parameters and graph..        ", end='', flush=True)
    start_time = time.time()
    compute(ctx.parameters, next(data)[0])
    print(f"Took {time.time() - start_time:.1f}s")

    parameters = ctx.parameters

    print("Compiling model..                       ", end='', flush=True)
    start_time = time.time()
    step(parameters, next(data))
    print(f"Took {time.time() - start_time:.1f}s")

    for name, param in parameters.items():
        print(name, util.prod(param.shape), param.shape)

    start_time = time.time()
    for idx, dat in enumerate(data):
        loss, parameters = step(parameters, dat)
        if idx % ctx.print_interval == 0:
            print(f'[{idx * ctx.device_steps:{len(str(ctx.steps * ctx.device_steps))}d}/{ctx.steps * ctx.device_steps}]'
                  f' Loss: {loss:6.3f} - Took: {time.time() - start_time:9.6f}s')
            start_time = time.time()


if __name__ == '__main__':
    main()
