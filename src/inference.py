import os
import typing

import jax
import numpy as np
from jax import lax, numpy as jnp, random

from src.context import Context, WhilePredictContext
from src.model import body_ctx, one_hot
from src.constants import ParallelAxes


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    return jnp.not_equal(jnp.mean(jnp.less(wctx.current_step, wctx.stop_pos)), 0)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    one_hot_mask = one_hot(wctx.current_step, wctx.ctx.dims.sizes.sequence)
    one_hot_mask = lax.broadcast(one_hot_mask, [wctx.ctx.dims.sizes.batch])

    out_token = body_ctx(wctx.ctx, wctx.data)

    key = random.PRNGKey(wctx.ctx.seed + wctx.current_step)
    temp = random.uniform(key, out_token.shape, maxval=1, minval=1e-7, dtype=jnp.float32)
    temp = jnp.log(temp)
    temp = jnp.negative(temp)
    temp = jnp.log(temp)
    temp = temp * lax.broadcast_in_dim(wctx.sampling_temperature, temp.shape, [0])

    sort = jnp.argsort(out_token)
    sort = one_hot(sort, out_token.shape[-1])
    sort = jnp.einsum("abcd,c->abd", sort, jnp.arange(out_token.shape[-1]))

    top_k_mask = jnp.less(sort, lax.broadcast_in_dim(wctx.top_k, sort.shape, [0]))

    out_token = out_token + temp
    out_token = out_token * top_k_mask
    out_token = jnp.argmax(out_token, -1)
    out_token = jnp.right_shift(out_token, jnp.array([1] * wctx.ctx.dims.sizes.batch)[0])

    one_hot_mask = one_hot_mask * jnp.greater_equal(wctx.current_step, wctx.start_pos)[0]
    wctx.data = wctx.data * (1 - one_hot_mask) + out_token * one_hot_mask

    wctx.current_step += 1

    return wctx.serialize()


def jitless_prediction_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray,
                            sampling_temperature: jnp.ndarray, top_k: jnp.ndarray, start_pos: jnp.ndarray,
                            stop_pos: jnp.ndarray) -> jnp.ndarray:
    wctx = WhilePredictContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx.sampling_temperature = sampling_temperature
    wctx.top_k = top_k
    wctx.start_pos = start_pos
    wctx.stop_pos = stop_pos
    wctx.current_step = jnp.min(start_pos)

    wctx = WhilePredictContext(lax.while_loop(cond_fn, body_fn, wctx.serialize()))

    return wctx.data


class Inference:
    def __init__(self, ctx: Context):
        ctx.initializing = True
        dummy_data = np.zeros((ctx.dims.sizes.batch, ctx.dims.sizes.sequence), dtype=np.int32)

        body_ctx(ctx, dummy_data)
        self.parameters = ctx.parameters

        partition = {'parameters': {k: 0 for k in ctx.parameters.keys()}}
        self.step = jax.pmap(jitless_prediction_step, axis_name=ParallelAxes.model,
                             in_axis=(partition, None, None, None, None, None),
                             out_axis=(None,))
        self.ctx = ctx

        self.complete_tokens(dummy_data, np.zeros((ctx.dims.sizes.batch,)), np.ones((ctx.dims.sizes.batch), ),
                             np.zeros((ctx.dims.sizes.batch,)), np.ones((ctx.dims.sizes.batch,)))

    def complete_tokens(self, prompt: np.array, sampling_temperature: np.array, top_k: np.array, start_pos: np.array,
                        stop_pos: np.array) -> np.array:
        return self.step(self.parameters, prompt, sampling_temperature, top_k, start_pos, stop_pos)

    def complete(self, text: str, sampling_temperature: float = 0.5, top_k: int = 32, length: int = 128):
        tokens = jnp.asarray(np.frombuffer(text.encode(), np.uint8)).astype(jnp.int32)
        base = jnp.zeros((self.ctx.dims.sizes.batch,))
        start = base + tokens.shape[0]
        out = self.complete_tokens(tokens, base + sampling_temperature, base + top_k, start, start + length)
        return np.asarray(out[start:]).tobytes().decode()


def main():
    model = Inference(Context())
    while True:
        prompt = input(">>> ")
        out = model.complete(prompt)
        print(out, "-" * os.get_terminal_size().columns, sep='\n')


if __name__ == '__main__':
    main()
