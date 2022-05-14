import os
import typing

import click
import jax
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from jax import lax, numpy as jnp, random
from pydantic import BaseModel
from transformers import GPT2TokenizerFast

from src.backend import matmul
from src.constants import ParallelAxes
from src.context import Context, WhilePredictContext
from src.main import get_parameters
from src.model import body_ctx, one_hot, promote_to
from src.utils.checkpoint import read_ckpt


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    is_eos = wctx.data == wctx.ctx.eval.eos
    behind_start = wctx.start_pos.reshape(-1, 1) > jnp.arange(wctx.ctx.dims.sequence).reshape(1, -1)
    is_eos = jnp.logical_and(is_eos, behind_start)
    is_eos = jnp.cumsum(is_eos, axis=1)
    eos_at_seq = (is_eos > 0).sum(0) == wctx.ctx.dims.batch
    eos = jnp.take_along_axis(eos_at_seq.reshape(-1), wctx.current_step.reshape(-1), axis=0)
    stop = jnp.less(wctx.current_step, wctx.stop_pos)
    return jnp.logical_or(eos, stop).reshape(())


def get_top_p_mask(out: jnp.ndarray, undo_indices: jnp.ndarray, top_p: jnp.ndarray) -> jnp.ndarray:
    cumulative_probabilities = lax.rev(jnp.cumsum(lax.rev(jax.nn.softmax(out), (1,)), -1), (1,))
    overflow = jnp.greater(cumulative_probabilities, top_p.reshape(-1, 1, 1))
    overflow = jnp.concatenate([overflow[:, :, 1:], jnp.zeros_like(overflow[:, :, :1])], -1)
    return jnp.take_along_axis(overflow, undo_indices, axis=2)


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    out, wgt = body_ctx(wctx.ctx, wctx.data)
    out = (out * one_hot(wctx.current_step - 1, wctx.ctx.dims.sequence).reshape(1, -1, 1)).sum(1, keepdims=True)
    out = matmul(out, wgt.transpose(1, 0)).reshape(out.shape[0], 1, -1)
    out = promote_to(out, jnp.float32)
    out_token = lax.psum(out, ParallelAxes.model)

    key = random.PRNGKey((wctx.ctx.seed + wctx.current_step).astype(jnp.int32))
    temp = random.uniform(key, out_token.shape, maxval=1, minval=1e-7, dtype=jnp.float32)
    temp = jnp.log(temp)
    temp = jnp.negative(temp)
    temp = jnp.log(temp)
    temp = temp * -wctx.temperature

    arange = lax.broadcasted_iota(jnp.int32, out_token.shape, dimension=2)
    sorted_out, argsort_out = lax.sort_key_val(out_token, arange)
    ranks = jnp.argsort(argsort_out, -1)
    top_k_mask = jnp.less(ranks, wctx.ctx.dims.vocab - wctx.top_k.reshape(-1, 1, 1))  # we want to not mask top k

    top_p_mask = get_top_p_mask(sorted_out, ranks, wctx.top_p)

    log_softmax = jax.nn.log_softmax(out_token)
    entropy = (jnp.exp(log_softmax) * log_softmax).sum(-1, keepdims=True)
    sorted_out, argsort_out = lax.sort_key_val(jnp.abs(entropy - log_softmax), arange)
    typical_mask = get_top_p_mask(sorted_out, jnp.argsort(argsort_out, -1), wctx.mass)

    out_token = out_token + temp + (top_k_mask + top_p_mask + typical_mask) * -1e9

    out_token = jnp.argmax(out_token, -1)
    wctx.data = jnp.where(one_hot(wctx.current_step, wctx.ctx.dims.sequence).reshape(1, -1), out_token, wctx.data)
    wctx.current_step += 1
    return wctx.serialize()


def jitless_prediction_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray,
                            temperature: jnp.ndarray, top_k: jnp.ndarray, top_p: jnp.ndarray, mass: jnp.ndarray,
                            seed: jnp.ndarray, start_pos: jnp.ndarray, stop_pos: jnp.ndarray) -> jnp.ndarray:
    wctx = WhilePredictContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx.temperature = temperature
    wctx.top_k = top_k
    wctx.top_p = top_p
    wctx.mass = mass
    wctx.ctx.seed = seed
    wctx.start_pos = start_pos
    wctx.stop_pos = stop_pos
    wctx.current_step = jnp.min(start_pos)

    wctx = WhilePredictContext(lax.while_loop(cond_fn, body_fn, wctx.serialize()))

    return wctx.data


class Inference:
    def __init__(self, ctx: Context):
        ctx.is_initializing = True
        dummy_data = np.zeros((1, ctx.dims.sequence), dtype=np.int32)
        get_parameters(ctx, dummy_data)
        read_ckpt(ctx)
        self.parameters = ctx.parameters

        ctx.is_initializing = False
        partition = {k: 0 for k in ctx.parameters.keys()}
        self.step = jax.pmap(jitless_prediction_step, axis_name=ParallelAxes.model,
                             in_axes=(partition, None, None, None, None, None, None, None, None), out_axes=None)
        self.ctx = ctx

        self.complete_jax(dummy_data, np.zeros(()), np.ones(()), np.ones(()), np.ones(()), np.ones(()),
                          np.zeros(()), np.ones(()))

    def complete_jax(self, prompt: jnp.array, temperature: jnp.array, top_k: jnp.array, top_p: jnp.array,
                     mass: jnp.ndarray, seed: jnp.array, start_pos: jnp.array, stop_pos: jnp.array) -> np.array:
        return self.step(self.parameters, prompt, temperature, top_k, top_p, mass, seed, start_pos, stop_pos)

    def complete_tokens(self, prompt: jnp.ndarray, temperature: float, top_k: int, top_p: float, mass: float, seed: int,
                        length: int) -> jnp.ndarray:
        tokens = jnp.pad(prompt, ((0, 0), (0, self.ctx.dims.sequence - prompt.shape[1])))
        base = jnp.zeros(())
        start = base + prompt.shape[1]
        return self.complete_jax(tokens, temperature, base + top_k, base + top_p, base + mass, base + seed, start,
                                 start + length)

    def complete(self, text: str, temperature: float = 0.5, top_k: int = 32, top_p: float = 0.9, mass: float = 0.9,
                 seed: int = 0, length: int = 128):
        tokens = jnp.asarray(np.frombuffer(text.encode(), np.uint8)).astype(jnp.int32).reshape(1, -1)
        out = self.complete_tokens(tokens, temperature, top_k, top_p, mass, seed, length)[0]
        return np.asarray(out).astype(np.uint8).tobytes().decode(errors='ignore')[len(text):len(text) + length]


class Tokens(BaseModel):
    tokens: typing.List[int]


class TokenCompletion(BaseModel):
    token_completion: typing.List[int]


class Completion(BaseModel):
    completion: str


class SanitizedTokens(BaseModel):
    tokens: typing.List[int]


class CompletionInput(BaseModel):
    prompt: str = ""
    length: int = 16
    temperature: float = 1.
    top_k: int = 64
    top_p: float = 0.9
    mass: float = 1
    seed: int = 0
    error: bool = True


class RestAPI:
    def __init__(self):
        self._ctx = Context()
        self._interface = Inference(self._ctx)
        if self._ctx.dims.vocab == 256:
            self._encode = lambda x: list(x.encode())
            self._decode = lambda x: np.asarray(x).astype(np.uint8).tobytes().decode(errors='ignore')
        else:
            tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
            self._encode = tokenizer.encode
            self._decode = tokenizer.decode

    async def check_tokens(self, tokens: typing.List[int], error: bool = True) -> SanitizedTokens:
        if tokens and max(tokens) > self._ctx.dims.vocab:
            if error:
                raise HTTPException(status_code=400, detail=f"Invalid tokens sent. Tokens go up to "
                                                            f"{self._ctx.dims.vocab} but received {max(tokens)}.")
            tokens = [t for t in tokens if t < self._ctx.dims.vocab]
        if len(tokens) > self._ctx.dims.sequence:
            if error:
                raise HTTPException(status_code=400, detail=f"Context too big. The model supports up to "
                                                            f"{self._ctx.dims.sequence} tokens but received "
                                                            f"{len(tokens)}.")
            tokens = tokens[:self._ctx.dims.sequence]
        return SanitizedTokens(tokens=tokens)

    async def encode(self, prompt: str) -> Tokens:
        return Tokens(tokens=self._encode(prompt))

    async def decode(self, prompt: typing.List[int]) -> Completion:
        return Completion(completion=self._decode(prompt))

    async def token_completion(self, params: CompletionInput) -> TokenCompletion:
        tokens = (await self.encode(params.prompt)).tokens
        tokens = (await self.check_tokens(tokens, params.error)).tokens
        tok = self._interface.complete_tokens(jnp.array(tokens).reshape(1, -1), params.temperature, params.top_k,
                                              params.top_p, params.mass, params.seed, params.length)
        tok = tok[0, len(tokens):len(tokens) + params.length].tolist()
        out = []
        for t in tok:
            if t == self._ctx.eval.eos:
                break
            out.append(t)
        return TokenCompletion(token_completion=out)

    async def completion(self, params: CompletionInput) -> Completion:
        return await self.decode((await self.token_completion(params)).token_completion)


@click.group()
def main():
    pass


@main.command()
def api():
    rest_api = RestAPI()
    fast_api = FastAPI()

    for key in dir(rest_api):
        if key.startswith('_') or key.endswith('_'):
            continue
        fn = getattr(rest_api, key)
        fast_api.post('/' + key, response_model=typing.get_type_hints(fn)["return"])(fn)

    uvicorn.run(fast_api, host='0.0.0.0', port=62220, log_level='info', workers=1)


@main.command()
@click.option('--temperature', default=0.5, type=float,
              help="Sampling temperature. Higher -> wider distribution / more random")
@click.option('--top-k', default=32, type=int, help="Across how many of the top tokens should be sampled")
@click.option('--top-p', default=1, type=float, help="How much probability mass to sample from")
@click.option('--length', default=128, type=int, help="Number of tokens to generate")
@click.option('--seed', default=0, type=int, help="Seed value for the random number generator")
def interactive(temperature: float, top_k: int, top_p: float, seed: int, length: int):
    model = Inference(Context())
    while True:
        prompt = input(">>> ")
        out = model.complete(prompt, temperature, top_k, top_p, seed, length)
        print(out, "-" * os.get_terminal_size().columns, sep='\n')


@main.command()
@click.option('--prompt', help="Text to feed into the language model.")
@click.option('--temperature', default=0.5, type=float,
              help="Sampling temperature. Higher -> wider distribution / more random")
@click.option('--top-k', default=32, type=int, help="Across how many of the top tokens should be sampled")
@click.option('--top-p', default=1, type=float, help="How much probability mass to sample from")
@click.option('--length', default=128, type=int, help="Number of tokens to generate")
@click.option('--seed', default=0, type=int, help="Seed value for the random number generator")
def once(prompt: str, temperature: float, top_k: int, top_p: float, seed: int, length: int):
    model = Inference(Context())
    out = model.complete(prompt, temperature, top_k, top_p, seed, length)
    print(out, "-" * os.get_terminal_size().columns, sep='\n')


if __name__ == '__main__':
    main()
