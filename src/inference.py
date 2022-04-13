import os
import typing

import jax
import numpy as np
from jax import lax, numpy as jnp, random

from src.constants import ParallelAxes
from src.context import Context, WhilePredictContext
from src.main import get_parameters
from src.model import body_ctx
from src.utils.checkpoint import read_ckpt


def one_hot(inp: jnp.ndarray, size: int) -> jnp.ndarray:
    return jnp.equal(jnp.reshape(inp, inp.shape + (1,)), jnp.reshape(jnp.arange(0, size), (1,) * inp.ndim + (size,)))


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    return jnp.not_equal(lax.pmean(jnp.less(wctx.current_step, wctx.stop_pos), ParallelAxes.model), 0)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    one_hot_mask = one_hot(wctx.current_step, wctx.ctx.dims.sizes.sequence)
    out_token = body_ctx(wctx.ctx, wctx.data)

    key = random.PRNGKey((wctx.ctx.seed + wctx.current_step).astype(jnp.int32))
    temp = random.uniform(key, out_token.shape, maxval=1, minval=1e-7, dtype=jnp.float32)
    temp = jnp.log(temp)
    temp = jnp.negative(temp)
    temp = jnp.log(temp)
    temp = temp * wctx.sampling_temperature

    sort = jnp.argsort(out_token)
    sort = one_hot(sort, out_token.shape[-1])
    sort = jnp.einsum("abcd,c->abd", sort, jnp.arange(out_token.shape[-1]))

    top_k_mask = jnp.less(sort, wctx.top_k)

    out_token = out_token + temp
    out_token = out_token * top_k_mask
    out_token = jnp.argmax(out_token, -1)
    out_token = jnp.right_shift(out_token, jnp.ones((1,), dtype=jnp.int32))

    one_hot_mask = one_hot_mask * jnp.greater_equal(wctx.current_step, wctx.start_pos)
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
        tokens = jnp.pad(tokens, ((0, 0), (0, self.ctx.dims.sizes.sequence - tokens.shape[-1])))
        base = jnp.zeros(())
        start = base + tokens.shape[0]
        out = self.complete_tokens(tokens, base + sampling_temperature, base + top_k, start, start + length)[0]
        return np.asarray(out).tobytes().decode()[len(text):]


def main():
    model = Inference(Context())
    while True:
        prompt = input(">>> ")
        out = model.complete(prompt)
        print(out, "-" * os.get_terminal_size().columns, sep='\n')


if __name__ == '__main__':
    main()
