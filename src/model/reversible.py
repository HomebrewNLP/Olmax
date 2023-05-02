import copy
from typing import Callable, Dict, Tuple, Union

import jax
from jax import numpy as jnp

from src.constants import SparseAccess
from src.context import Context

REVERSIBLE_CTX = Tuple[Dict[str, jax.Array], jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]
Output = Union[jax.Array, Tuple[jax.Array, jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]
ReversibleFn = Callable[[Context, jax.Array, jax.Array, jax.Array], Output]
FourArrays = Tuple[jax.Array, jax.Array, jax.Array, jax.Array]


def _reversible_at_init(ctx: Context, fn: ReversibleFn, sparse_access: SparseAccess, src: REVERSIBLE_CTX,
                        *args) -> REVERSIBLE_CTX:
    params, _x00, x01, x10, x11, sparse, d_sparse = src
    new_ctx = ctx.add_to_prefix("reversible")
    new_ctx.parameters = params
    out = fn(new_ctx, x10, *(sparse,) * (sparse_access == SparseAccess.read), *args)
    if sparse_access in (SparseAccess.read, SparseAccess.write):
        out, *_ = out
    ctx.parameters = new_ctx.parameters
    ctx.name_cache = new_ctx.name_cache
    ctx.prng_key = new_ctx.prng_key
    return new_ctx.parameters, x10, x11, out, x01, sparse, d_sparse


def reversible(ctx: Context, fn: ReversibleFn, sparse_access: SparseAccess, src: REVERSIBLE_CTX,
               *args) -> REVERSIBLE_CTX:
    if ctx.is_initializing:
        return _reversible_at_init(ctx, fn, sparse_access, src, *args)

    name_cache = copy.deepcopy(ctx.name_cache)

    def base(params: Dict[str, jax.Array], inp: jax.Array, *inner_args) -> Output:
        ctx.name_cache = copy.deepcopy(name_cache)
        new_ctx = ctx.add_to_prefix("reversible")
        new_ctx.parameters = params
        out = fn(new_ctx, inp, *inner_args)
        ctx.name_cache = new_ctx.name_cache
        return out

    @jax.custom_gradient
    def _fn(inputs: REVERSIBLE_CTX, *inner_args: jax.Array):
        params, x0, _dx0, x1, _dx1, sparse, d_sparse = inputs

        def _grad(dy):
            d_params_old, dy0, y0, dy1, y1, dy_sparse, y_sparse = dy
            x0, grad_fn = jax.vjp(base, params, y0, *(y_sparse,) * (sparse_access == SparseAccess.read), *inner_args)
            if sparse_access == SparseAccess.write:
                x0, keys, vals = x0
                y_sparse = y_sparse.at[jnp.arange(keys.size) // ctx.dims.memory_slots, keys].sub(vals)
                sparse_items = jnp.take_along_axis(dy_sparse, keys, 1).reshape(ctx.dims.batch, -1)
                d_params, dx0, *_ = grad_fn(dy1, None, sparse_items)
            elif sparse_access == SparseAccess.read:
                d_params, dx0, dsparse, *_ = grad_fn(dy1, None)
                x0, keys = x0
                dy_sparse = dy_sparse.at[jnp.arange(keys.size) // ctx.dims.memory_slots, keys].add(dsparse)
            else:
                d_params, dx0, *_ = grad_fn(dy1)
                dx0, *_ = dx0
            d_params = {k: d_params_old.get(k, 0) + d_params.get(k, 0) for k in d_params.keys()}
            return (d_params, dy1, y1 - x0, dx0 + dy0, y0, dy_sparse, y_sparse), (None,) * len(inner_args)

        out = base(params, x1, *(sparse,) * (sparse_access == SparseAccess.read), *inner_args)
        if sparse_access == SparseAccess.write:
            out, vals, keys = out
            sparse = sparse.at[jnp.arange(keys.size) // ctx.dims.memory_slots, keys].add(vals)
        out = x0 + out
        return (params, x1, x1, out, out, sparse, sparse), _grad

    return _fn(src, args)


def revnet_out(src: FourArrays) -> jax.Array:
    @jax.custom_gradient
    def _fn(x0: jax.Array, _x0_back: jax.Array, x1: jax.Array, _x1_back: jax.Array):
        def _grad(dy) -> FourArrays:
            return dy, x0, dy, x1

        return x0 + x1, _grad

    return _fn(*src)
