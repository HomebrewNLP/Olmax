import os
import typing

import jax
import numpy as np
from jax import lax, numpy as jnp, random

from src.backend import matmul
from src.constants import ParallelAxes
from src.context import Context, WhilePredictContext
from src.main import get_parameters
from src.model import body_ctx, one_hot
from src.utils.checkpoint import read_ckpt


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    return jnp.less(wctx.current_step, wctx.stop_pos)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    out, wgt = body_ctx(wctx.ctx, wctx.data)
    out = (out * one_hot(wctx.current_step - 1, wctx.ctx.dims.sizes.sequence).reshape(1, -1, 1)).sum(1, keepdims=True)
    out_token = matmul(out, wgt.transpose(1, 0)).reshape(out.shape[0], 1, -1)

    key = random.PRNGKey((wctx.ctx.seed + wctx.current_step).astype(jnp.int32))
    temp = random.uniform(key, out_token.shape, maxval=1, minval=1e-7, dtype=jnp.float32)
    temp = jnp.log(temp)
    temp = jnp.negative(temp)
    temp = jnp.log(temp)
    temp = temp * -wctx.sampling_temperature

    # sort = jnp.argsort(out_token)
    # sort = one_hot(sort, out_token.shape[-1])
    # sort = jnp.einsum("abcd,c->abd", sort, jnp.arange(out_token.shape[-1]))
    # top_k_mask = jnp.greater_equal(sort, wctx.top_k)

    out_token = out_token + temp
    # out_token = out_token + top_k_mask * -1e9

    out_token = jnp.argmax(out_token, -1)
    wctx.data = jnp.where(one_hot(wctx.current_step, wctx.ctx.dims.sizes.sequence).reshape(1, -1), out_token, wctx.data)
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
        ctx.is_initializing = True
        dummy_data = np.zeros((1, ctx.dims.sizes.sequence), dtype=np.int32)
        get_parameters(ctx, dummy_data)
        read_ckpt(ctx)
        self.parameters = ctx.parameters

        ctx.is_initializing = False
        partition = {k: 0 for k in ctx.parameters.keys()}
        self.step = jax.pmap(jitless_prediction_step, axis_name=ParallelAxes.model,
                             in_axes=(partition, None, None, None, None, None), out_axes=None)
        self.ctx = ctx

        self.complete_tokens(dummy_data, np.zeros(()), np.ones(()), np.zeros(()), np.ones(()))

    def complete_tokens(self, prompt: np.array, sampling_temperature: np.array, top_k: np.array, start_pos: np.array,
                        stop_pos: np.array) -> np.array:
        return self.step(self.parameters, prompt, sampling_temperature, top_k, start_pos, stop_pos)

    def complete(self, text: str, sampling_temperature: float = 0.5, top_k: int = 32, length: int = 128):
        tokens = jnp.asarray(np.frombuffer(text.encode(), np.uint8)).astype(jnp.int32).reshape(1, -1)
        tokens = jnp.pad(tokens, ((0, 0), (0, self.ctx.dims.sizes.sequence - len(text))))
        base = jnp.zeros(())
        start = base + len(text)
        out = self.complete_tokens(tokens, base + sampling_temperature, base + top_k, start, start + length)[0]
        return np.asarray(out).astype(np.uint8).tobytes().decode(errors='ignore')[len(text):len(text) + length]


def main():
    model = Inference(Context())
    while True:
        prompt = input(">>> ")
        out = model.complete(prompt)
        print(out, "-" * os.get_terminal_size().columns, sep='\n')


if __name__ == '__main__':
    main()
