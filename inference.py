import typing

import jax
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from jax import lax, numpy as jnp, random
from pydantic import BaseModel
from transformers import GPT2TokenizerFast

from src.backend import matmul, promote_to
from src.constants import ParallelAxes
from src.context import Context, WhilePredictContext
from src.main import get_parameters
from src.model.main import body_ctx
from src.model.moe import one_hot
from src.utils.checkpoint import read_ckpt


def cond_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> bool:
    wctx = WhilePredictContext(while_ctx_dict)
    is_eos = wctx.data == wctx.ctx.eval.eos
    behind_start = wctx.start_pos.reshape(-1, 1) > jnp.arange(wctx.ctx.dims.sequence).reshape(1, -1)
    is_eos = jnp.logical_and(is_eos, behind_start)
    is_eos = jnp.cumsum(is_eos, axis=1)
    eos_at_seq = (is_eos > 0).sum(0) == wctx.ctx.dims.batch
    eos = jnp.take_along_axis(eos_at_seq.reshape(-1), wctx.current_step.reshape(-1).astype(jnp.int32), axis=0)
    stop = jnp.less(wctx.current_step, wctx.stop_pos)
    return jnp.logical_or(eos, stop).reshape(())


def body_fn(while_ctx_dict: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    wctx = WhilePredictContext(while_ctx_dict)

    out, wgt = body_ctx(wctx.ctx, wctx.data)
    out = (out * one_hot(wctx.current_step - 1, wctx.ctx.dims.sequence).reshape(1, -1, 1)).sum(1, keepdims=True)
    out = matmul(out, wgt).reshape(out.shape[0], 1, -1)
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
    top_k_mask = jnp.less(ranks, wctx.ctx.dims.vocab - wctx.max_tokens.reshape(-1, 1, 1))  # we want to not mask top k

    cumulative_probabilities = lax.rev(jnp.cumsum(lax.rev(jax.nn.softmax(out), (1,)), -1), (1,))
    overflow = jnp.greater(cumulative_probabilities, wctx.max_probability_mass.reshape(-1, 1, 1))
    overflow = jnp.concatenate([overflow[:, :, 1:], jnp.zeros_like(overflow[:, :, :1])], -1)
    top_p_mask = jnp.take_along_axis(overflow, ranks, axis=2)

    log_softmax = jax.nn.log_softmax(out_token)
    shifted_scores = jnp.abs((jnp.exp(log_softmax) * log_softmax).sum(-1, keepdims=True) - log_softmax)
    sorted_out, argsort_out = lax.sort_key_val(shifted_scores, arange)
    cumulative_probabilities = jnp.cumsum(jax.nn.softmax(jnp.take_along_axis(out_token, argsort_out, axis=2)), -1)
    overflow = jnp.less(cumulative_probabilities, wctx.typical_mass.reshape(-1, 1, 1))
    overflow_at = overflow.sum(-1, keepdims=True).astype(jnp.int32)
    overflow = jnp.take_along_axis(sorted_out, overflow_at, axis=2)
    overflow = jnp.greater(sorted_out, overflow)
    overflow = jnp.concatenate([jnp.zeros_like(overflow[:, :, :1]), overflow[:, :, :-1]], -1)
    typical_mask = jnp.take_along_axis(overflow, jnp.argsort(argsort_out, -1), axis=2)

    # min_prob_mask ("top-p-x") and adaptive_mask ("top-a") are ideas taken from
    # https://github.com/BlinkDL/RWKV-LM/blob/4bbee4bb1a26059c6425d25c59e057891ae7c4c7/README.md
    softmax = jax.nn.softmax(out_token)
    min_prob_mask = softmax < wctx.max_probability_to_filter.reshape(-1, 1, 1)
    adaptive_filter = jnp.max(softmax, axis=2, keepdims=True) ** wctx.adaptive_filter_power * wctx.adaptive_filter_scale
    adaptive_mask = softmax < adaptive_filter

    out_token = out_token + temp + ((top_k_mask + top_p_mask + adaptive_mask) * min_prob_mask + typical_mask) * -1e9
    out_token = jnp.argmax(out_token, -1)
    wctx.data = jnp.where(one_hot(wctx.current_step, wctx.ctx.dims.sequence).reshape(1, -1), out_token, wctx.data)
    wctx.current_step += 1
    return wctx.serialize()


def jitless_prediction_step(parameters: typing.Dict[str, jnp.ndarray], data: jnp.ndarray,
                            temperature: jnp.ndarray, max_tokens: jnp.ndarray, max_probability_mass: jnp.ndarray,
                            typical_mass: jnp.ndarray, max_probability_to_filter: jnp.ndarray,
                            adaptive_filter_power: jnp.ndarray, adaptive_filter_scale: jnp.ndarray, seed: jnp.ndarray,
                            start_pos: jnp.ndarray, stop_pos: jnp.ndarray) -> jnp.ndarray:
    wctx = WhilePredictContext()
    wctx.ctx.parameters = parameters
    wctx.data = data
    wctx.temperature = temperature
    wctx.max_tokens = max_tokens
    wctx.max_probability_to_filter = max_probability_to_filter
    wctx.max_probability_mass = max_probability_mass
    wctx.adaptive_filter_power = adaptive_filter_power
    wctx.adaptive_filter_scale = adaptive_filter_scale
    wctx.typical_mass = typical_mass
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
                             in_axes=(partition, None, None, None, None, None, None, None, None, None, None, None),
                             out_axes=None)
        self.ctx = ctx

        self.complete_jax(dummy_data, np.zeros(()), np.ones(()), np.ones(()), np.ones(()), np.ones(()), np.ones(()),
                          np.ones(()), np.zeros(()), np.zeros(()), np.ones(()))

    def complete_jax(self, prompt: jnp.array, temperature: jnp.array, max_tokens: jnp.array,
                     max_probability_mass: jnp.array, typical_mass: jnp.ndarray,
                     max_probability_to_filter: jnp.ndarray, adaptive_filter_power: jnp.ndarray,
                     adaptive_filter_scale: jnp.ndarray, seed: jnp.array, start_pos: jnp.array,
                     stop_pos: jnp.array) -> jnp.array:
        return self.step(self.parameters, prompt, temperature, max_tokens, max_probability_mass, typical_mass,
                         max_probability_to_filter, adaptive_filter_power, adaptive_filter_scale, seed, start_pos,
                         stop_pos)

    def complete_tokens(self, prompt: jnp.ndarray, temperature: float, max_tokens: int, max_probability_mass: float,
                        typical_mass: float, max_probability_to_filter: float, adaptive_filter_power: float,
                        adaptive_filter_scale: float, seed: int, length: int) -> jnp.ndarray:
        tokens = jnp.pad(prompt, ((0, 0), (0, self.ctx.dims.sequence - prompt.shape[1])))
        base = jnp.zeros(())
        start = base + prompt.shape[1]
        return self.complete_jax(tokens, temperature, base + max_tokens, base + max_probability_mass,
                                 base + typical_mass, base + max_probability_to_filter, base + adaptive_filter_power,
                                 base + adaptive_filter_scale, base + seed, start, start + length)

    def complete(self, text: str, temperature: float = 0.5, max_tokens: int = 32, max_probability_mass: float = 0.9,
                 typical_mass: float = 1, max_probability_to_filter: float = 1., adaptive_filter_power: float = 1,
                 adaptive_filter_scale: float = 0, seed: int = 0, length: int = 128):
        tokens = jnp.asarray(np.frombuffer(text.encode(), np.uint8)).astype(jnp.int32).reshape(1, -1)
        out = self.complete_tokens(tokens, temperature, max_tokens, max_probability_mass, typical_mass,
                                   max_probability_to_filter, adaptive_filter_power, adaptive_filter_scale, seed,
                                   length)[0]
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
    max_tokens: int = 64
    max_probability_mass: float = 0.9
    typical_mass: float = 1
    max_probability_to_filter: float = 1
    adaptive_filter_power: float = 1
    adaptive_filter_scale: float = 0
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
        tok = self._interface.complete_tokens(jnp.array(tokens).reshape(1, -1), params.temperature, params.max_tokens,
                                              params.max_probability_mass, params.typical_mass,
                                              params.max_probability_to_filter, params.adaptive_filter_power,
                                              params.adaptive_filter_scale, params.seed, params.length)
        tok = tok[0, len(tokens):len(tokens) + params.length].tolist()
        out = []
        for t in tok:
            if t == self._ctx.eval.eos:
                break
            out.append(t)
        return TokenCompletion(token_completion=out)

    async def completion(self, params: CompletionInput) -> Completion:
        return await self.decode((await self.token_completion(params)).token_completion)


def main():
    rest_api = RestAPI()
    fast_api = FastAPI()

    for key in dir(rest_api):
        if key.startswith('_') or key.endswith('_'):
            continue
        fn = getattr(rest_api, key)
        fast_api.post('/' + key, response_model=typing.get_type_hints(fn)["return"])(fn)

    uvicorn.run(fast_api, host='0.0.0.0', port=62220, log_level='info', workers=1)


if __name__ == '__main__':
    main()
