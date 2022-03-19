import typing

import jax
import numpy as np
from jax import lax, numpy as jnp, random
from jax.experimental import PartitionSpec
from jax.experimental import pjit
from jax.experimental.maps import mesh

from src.main import sharding
from src.context import Context, WhilePredictContext
from src.model import body_ctx, one_hot


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    return jnp.not_equal(jnp.mean(jnp.less(wctx.current_step, wctx.stop_pos)), 0)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    one_hot_mask = one_hot(wctx.current_step, wctx.ctx.dims.dim_sizes[wctx.ctx.dims.sequence])
    one_hot_mask = lax.broadcast(one_hot_mask, [wctx.ctx.dims.dim_sizes[wctx.ctx.dims.batch]])

    out_token = body_ctx(wctx.ctx, wctx.data)

    temp = random.uniform(random.PRNGKey(wctx.ctx.seed), out_token.shape, maxval=1, minval=1e-9, dtype=jnp.float32)
    temp = jnp.log(temp)
    temp = jnp.negative(temp)
    temp = jnp.log(temp)
    temp = temp * lax.broadcast_in_dim(wctx.sampling_temperature, temp.shape, [0])

    sort = jnp.argsort(out_token)
    sort = one_hot(sort, out_token.shape[-1])
    sort = jnp.einsum("abcd,c->abd", sort, jnp.arange(out_token.shape[-1]))

    top_n_mask = jnp.less(sort, lax.broadcast_in_dim(wctx.top_n, sort.shape, [0]))

    out_token = out_token + temp
    out_token = out_token * top_n_mask
    out_token = jnp.argmax(out_token, -1)
    out_token = jnp.right_shift(out_token, jnp.array([1] * wctx.ctx.dims.dim_sizes[wctx.ctx.dims.batch])[0])

    one_hot_mask = one_hot_mask * jnp.greater_equal(wctx.current_step, wctx.start_pos)[0]
    wctx.data = wctx.data * (1 - one_hot_mask) + out_token * one_hot_mask
    wctx.data = lax.convert_element_type(wctx.data, jnp.float32)

    wctx.current_step += 1

    return wctx.serialize()


def jitless_prediction_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray,
                            sampling_temperature: jnp.ndarray,
                            top_n: jnp.ndarray,
                            start_pos: jnp.ndarray,
                            stop_pos: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    wctx = WhilePredictContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx.sampling_temperature = sampling_temperature
    wctx.top_n = top_n
    wctx.start_pos = start_pos
    wctx.stop_pos = stop_pos
    wctx.current_step = jnp.min(start_pos)

    wctx = WhilePredictContext(lax.while_loop(cond_fn, body_fn, wctx.serialize()))

    return wctx.data, wctx.ctx.parameters


class Infrerence_Model():
    def __init__(self, ctx: Context):
        ctx.initializing = True
        dumy_data = np.zeros((ctx.dims.dim_sizes[ctx.dims.batch],
                              ctx.dims.dim_sizes[ctx.dims.sequence]), dtype=np.float)

        body_ctx(ctx, dumy_data)
        self.parameters = ctx.parameters

        partition = {name: sharding(ctx, dims) for name, dims in ctx.parameter_dims.items()}
        self.step = pjit.pjit(jitless_prediction_step,
                              in_axis_resources=(
                                  partition, PartitionSpec("data_parallel", None), None, None, None, None),
                              out_axis_resources=(PartitionSpec(None, None), partition))
        self.mesh_devices = np.array(jax.devices()).reshape(ctx.data_parallel, ctx.model_parallel)

        self.compleat(dumy_data,
                      np.zeros((ctx.dims.dim_sizes[ctx.dims.batch])),
                      np.ones((ctx.dims.dim_sizes[ctx.dims.batch])),
                      np.zeros((ctx.dims.dim_sizes[ctx.dims.batch])),
                      np.array([ctx.dims.dim_sizes[ctx.dims.batch]] * ctx.dims.dim_sizes[ctx.dims.vocab]))

    def compleat(self, promt: np.array,
                 sampling_temperature: np.array,
                 top_n: np.array,
                 start_pos: np.array,
                 stop_pos: np.array) -> np.array:
        with mesh(self.mesh_devices, ('data_parallel', 'model_parallel')):
            out, _ = self.step(self.parameters, promt, sampling_temperature, top_n, start_pos, stop_pos)

        return out


if __name__ == '__main__':
    model = Infrerence_Model(Context())
