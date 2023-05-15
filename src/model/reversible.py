import copy
from typing import Callable, Dict, Tuple, Union, List

import jax
from jax import numpy as jnp

from src.constants import SparseAccess
from src.context import Context

REVERSIBLE_CTX = Tuple[jax.Array, jax.Array]
FourArrays = Tuple[jax.Array, jax.Array, jax.Array, jax.Array]
Output = Union[jax.Array, Union[FourArrays, Tuple[jax.Array, jax.Array, jax.Array]]]
ReversibleFn = Callable[[Context, jax.Array, jax.Array, jax.Array], Output]


def _reversible_at_init(ctx: Context, fn: ReversibleFn, sparse_access: SparseAccess, src: REVERSIBLE_CTX,
                        *args) -> REVERSIBLE_CTX:
    x, sparse = src
    new_ctx = ctx.add_to_prefix("reversible")
    out = fn(new_ctx, x, *(sparse,) * (sparse_access == SparseAccess.read), *args)
    if sparse_access in (SparseAccess.read, SparseAccess.write):
        out, *_ = out
    ctx.parameters = new_ctx.parameters
    ctx.name_cache = new_ctx.name_cache
    ctx.prng_key = new_ctx.prng_key
    return x, sparse


def at_sparse(sparse, keys):
    batch_size, memory_slots = keys.shape
    return sparse.at[jnp.repeat(jnp.arange(batch_size)[:, None], memory_slots, 1), keys]


def reversible(ctx: Context, fn: ReversibleFn, sparse_access: SparseAccess, src: REVERSIBLE_CTX,
               *args: jax.Array) -> REVERSIBLE_CTX:
    if ctx.is_initializing:
        return _reversible_at_init(ctx, fn, sparse_access, src, *args)

    name_cache = copy.deepcopy(ctx.name_cache)

    def base(params: Dict[str, jax.Array], inp: jax.Array, inner_args) -> Output:
        ctx.name_cache = copy.deepcopy(name_cache)
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, inp, *inner_args)
        ctx.name_cache = new_ctx.name_cache
        if sparse_access in (SparseAccess.write, SparseAccess.read):
            return tuple(out[:-1]), out[-1]
        return out

    @jax.custom_gradient
    def _fn(params: Dict[str, jax.Array], inputs: REVERSIBLE_CTX, inner_args: List[jax.Array]):
        x, sparse = inputs

        def _grad(dy):
            dy, dy_sparse = dy
            out, grad_fn, *keys = jax.vjp(base, params, x,
                                          [*(sparse,) * (sparse_access == SparseAccess.read), *inner_args],
                                          has_aux=sparse_access in (SparseAccess.read, SparseAccess.write))
            if sparse_access in (SparseAccess.read, SparseAccess.write):
                keys = keys[0]
            if sparse_access == SparseAccess.write:
                _, vals = out
                sparse_items = jnp.take_along_axis(dy_sparse, keys.reshape(*keys.shape, 1), 1).reshape(*vals.shape)
                d_params, dx, _ = grad_fn((dy, sparse_items))
            elif sparse_access == SparseAccess.read:
                d_params, dx, (dsparse, *_) = grad_fn((dy,))
                dy_sparse = dy_sparse + dsparse  # TODO: Find a way to get the sparse gradients and scatter-add manually
            else:
                d_params, dx, _ = grad_fn(dy)
            return d_params, (dy + dx, dy_sparse), [jnp.zeros_like(a) for a in args]

        out = base(params, x, [*(sparse,) * (sparse_access == SparseAccess.read), *inner_args])
        if sparse_access == SparseAccess.write:
            (out, vals), keys = out
            sparse = at_sparse(sparse, keys).add(vals)
        elif sparse_access == SparseAccess.read:
            (out,), keys = out
        return (x + out, sparse), _grad

    return _fn(ctx.parameters, src, list(args))
