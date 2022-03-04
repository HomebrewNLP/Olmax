"""
Adapted from https://github.com/kingoflolz/mesh-transformer-jax/blob/0a75ca9370576ad9d247facf6cb8e9699300e690/mesh_transformer/checkpoint.py
"""

import functools
import io
import multiprocessing
import time

import jax
import jax.numpy as jnp
import numpy as np
from smart_open import open

from src.context import Context

pieces = 16  # how many files to split each shard across


@functools.partial(jax.jit, backend="cpu")
def index_weights(weights, idx):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jax.tree_map(lambda i: i[idx], weights), cpu_device)


def write(x, ckpt_dir):
    idx, i = x
    file_path = ckpt_dir + f"{idx}.npz"
    for _ in range(3):
        try:
            with open(file_path, "wb") as f:
                np.savez(f, *i)
            return
        except:
            print("save failed, trying again")

    print("save failed 3 times, exiting")
    raise Exception("save failed")


def write_ckpt(ctx: Context):
    flattened, structure = jax.tree_flatten(ctx.parameters)

    for shard in range(ctx.dims.sizes.heads):
        cpu_flattened = index_weights(flattened, shard)

        k, m = divmod(len(cpu_flattened), pieces)
        cpu_flattened_chunked = (cpu_flattened[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(pieces))

        with multiprocessing.pool.ThreadPool(pieces) as p:
            write_fn = functools.partial(write, ckpt_dir=f"{ctx.training.checkpoint_path}/")
            list((p.imap_unordered(write_fn, enumerate(cpu_flattened_chunked))))


def read_shard(ckpt_dir):
    out = []
    for idx in range(16):
        file_path = ckpt_dir + f"{idx}.npz"
        with open(file_path, "rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            deserialized = np.load(f_io)
            for i in deserialized:
                out.append(deserialized[i])
    return out


def read_ckpt(pytree, path, shards_in, load_opt=True):
    old_flattened, structure = jax.tree_flatten(pytree)

    original_opt_state = pytree["opt_state"]

    with multiprocessing.pool.ThreadPool(shards_in) as p:
        start = time.time()
        shards = list((p.imap(read_shard, [f"{path}shard_{i}/" for i in range(shards_in)])))
        print(f"read from disk/gcs in {time.time() - start:.06}s")

    def _unshard(shards, old_flattened):
        unsharded = []

        for old, *all_shards in zip(old_flattened, *shards):
            x = np.stack(all_shards)
            # No idea why this is V2...?
            if x.dtype == np.dtype('V2'):
                x.dtype = jnp.bfloat16

            unsharded.append(x)

            assert x.shape == old.shape, f"Incompatible checkpoints {x.shape} vs {old.shape}"
        return unsharded

    try:
        unsharded = _unshard(shards, old_flattened)
    except AssertionError:
        load_opt = False  # no opt to load in ckpt
        del pytree['opt_state']
        old_flattened, structure = jax.tree_flatten(pytree)
        unsharded = _unshard(shards, old_flattened)

    loaded_pytree = jax.tree_unflatten(structure, unsharded)

    if not load_opt:
        loaded_pytree['opt_state'] = original_opt_state
    return loaded_pytree
