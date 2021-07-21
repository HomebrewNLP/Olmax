import math
import time
import typing
import warnings

import jax
import jax._src.util as util
import numpy as np
from jax import lax, numpy as jnp, random
from jax.experimental import PartitionSpec
from jax.experimental import pjit
from jax.experimental.maps import mesh

from context import Context, WhilePredictContext
from data import text_dataset

from model import shard, body_ctx, sharding, one_hot



def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    return jnp.not_equal(jnp.mean(jnp.less(wctx.current_step, wctx.stop_pos)), 0)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    one_hot_mask = one_hot(wctx.current_step, wctx.ctx.dims.dim_sizes[wctx.ctx.dims.sequence])
    one_hot_mask = lax.broadcast(one_hot_mask, [wctx.ctx.dims.dim_sizes[wctx.ctx.dims.batch]])

    out = body_ctx(wctx.ctx, wctx.data)
    out_token = jnp.argmax(out, -1)
    out_token = jnp.right_shift(out_token, jnp.array([1] * wctx.ctx.dims.dim_sizes[wctx.ctx.dims.batch])[0])

    one_hot_mask = one_hot_mask * jnp.greater_equal(wctx.current_step, wctx.start_pos)[0]
    wctx.data = wctx.data * (1 - one_hot_mask) + out_token * one_hot_mask
    wctx.data = lax.convert_element_type(wctx.data, jnp.float32)

    wctx.current_step += 1

    return wctx.serialize()


def jitless_prediction_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray) -> typing.Tuple[
    jnp.ndarray, typing.Dict[str, jnp.ndarray]]:
    wctx = WhilePredictContext()
    wctx.ctx.parameters = parameters
    wctx.data = data

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
                         in_axis_resources=(partition, PartitionSpec("data_parallel", None)),
                         out_axis_resources=(PartitionSpec(None, None), partition))
        self.mesh_devices = np.array(jax.devices()).reshape(ctx.data_parallel, ctx.model_parallel)

        self.compleat(dumy_data)


    def compleat(self, promt):
        with mesh(self.mesh_devices, ('data_parallel', 'model_parallel')):
            out, _ = self.step(self.parameters, promt)

        print(out, out.shape)


if __name__ == '__main__':
    model = Infrerence_Model(Context())