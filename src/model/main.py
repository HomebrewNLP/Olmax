from typing import Tuple, Dict, Optional

import jax

from src.backend import with_context
from src.constants import SparseAccess
from src.context import Context
from src.model.linear import read, write
from src.model.loss import loss_fn
from src.model.reversible import reversible


@with_context()
def body_ctx(ctx: Context, src: Tuple[jax.Array, jax.Array], inp: jax.Array, tgt: jax.Array) -> Optional[
    Tuple[jax.Array, jax.Array]]:
    for i in range(ctx.dims.depth):
        src = reversible(ctx, read, SparseAccess.read, src, inp)
        src = reversible(ctx, write, SparseAccess.write, src, inp)
    loss = loss_fn(ctx, src[0], tgt)
    if ctx.is_initializing:
        return
    return loss[0], (src, loss[1:])


def compute(params: Dict[str, jax.Array], carry: Tuple[jax.Array, jax.Array], inp: jax.Array
            ) -> Optional[Tuple[jax.Array, jax.Array]]:
    ctx = Context()
    ctx.parameters = params
    src, tgt = inp
    return body_ctx(ctx, carry, src, tgt)
