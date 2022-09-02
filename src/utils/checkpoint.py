"""
Adapted from https://github.com/kingoflolz/mesh-transformer-jax/blob/0a75ca9370576ad9d247facf6cb8e9699300e690
/mesh_transformer/checkpoint.py
"""
import functools
import io
import json
import multiprocessing
import re
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from smart_open import open as smart_open

from src.backend import is_main
from src.context import Context

WEIGHT_PIECES = 16  # how many files to split each shard across
UPLOAD_RETRIES = 8


@functools.partial(jax.jit, backend="cpu")
def index_weights(weights, idx):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jax.tree_util.tree_map(lambda i: i[idx], weights), cpu_device)


def write(x, ckpt_dir):
    idx, i = x
    file_path = ckpt_dir + f"{idx}.npz"
    for _ in range(UPLOAD_RETRIES):
        try:
            with smart_open(file_path, "wb") as f:
                np.savez(f, *i)
            return
        except:  # skipcq: FLK-E722
            print("save failed, trying again")

    print("save failed 3 times, exiting")
    raise Exception("save failed")


def write_ckpt(ctx: Context):
    flattened, structure = jax.tree_util.tree_flatten(ctx.parameters)

    structure = str(structure)  # like "PyTreeDef({'2': {'a': *}})"
    structure = structure.replace('PyTreeDef', '')[1:-1]  # clean up "types"
    structure = structure.replace(': *', ': null').replace("{'", '{"').replace("':", '":')
    structure = structure.replace("', ", '", ').replace(", '", ', "')  # to valid JSON

    if is_main():
        success = False
        for _ in range(UPLOAD_RETRIES):
            try:
                with smart_open(f"{ctx.training.checkpoint_path}/structure.json", "w") as f:  # skipcq: PTC-W6004
                    f.write(structure)
            except:  # skipcq: FLK-E722
                print("Failed to save structure. Traceback:")
                traceback.print_exc()
                continue
            success = True
            break
        if not success:
            raise ValueError("Couldn't save structure")

    for device in jax.local_devices():
        shard = device.id
        cpu_flattened = index_weights(flattened, shard)

        k, m = divmod(len(cpu_flattened), WEIGHT_PIECES)
        cpu_flattened_chunked = (cpu_flattened[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in
                                 range(WEIGHT_PIECES))

        with multiprocessing.pool.ThreadPool(WEIGHT_PIECES) as p:
            write_fn = functools.partial(write, ckpt_dir=f"{ctx.training.checkpoint_path}/{shard}_")
            list((p.imap_unordered(write_fn, enumerate(cpu_flattened_chunked))))


def read_shard(ckpt_dir):
    out = []
    for idx in range(WEIGHT_PIECES):
        file_path = ckpt_dir + f"{idx}.npz"
        with smart_open(file_path, "rb") as f:
            buf = f.read()
            f_io = io.BytesIO(buf)
            deserialized = np.load(f_io)
            for i in deserialized:
                out.append(deserialized[i])
    return out


def deep_replace(d, value):
    if isinstance(d, dict):
        return {k: deep_replace(v, value) for k, v in d.items()}
    return value


def depth(param_name, name: str = 'reversible'):
    return int(param_name.split(f'/{name}:')[1].split('/')[0])


def read_ckpt(ctx: Context, ignore: str = '.*optimizer.*'):
    ignore = re.compile(ignore)

    with smart_open(f"{ctx.training.checkpoint_load_path}/structure.json", "r") as f:
        new_structure = f.read()
    new_structure = json.loads(new_structure)
    new_structure = deep_replace(new_structure, jnp.zeros((1,)))
    _, new_structure = jax.tree_util.tree_flatten(new_structure)

    with multiprocessing.pool.ThreadPool(ctx.dims.heads) as p:
        start = time.time()
        shards = list(p.imap(read_shard, [f"{ctx.training.checkpoint_load_path}/{i}_" for i in range(ctx.dims.heads)]))
        print(f"read from disk/gcs in {time.time() - start:.06}s")

    unsharded = []
    for all_shards in zip(*shards):
        x = np.stack(all_shards)
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        unsharded.append(x)
    params = jax.tree_util.tree_unflatten(new_structure, unsharded)

    print("Unknown parameters:  ", [p for p in params.keys() if p not in ctx.parameters and not ignore.match(p)])
    print("Unfilled parameters: ", [p for p in ctx.parameters.keys() if p not in params and not ignore.match(p)])

    if not ctx.parameters:
        for key, param in params.items():
            if key in ctx.parameters:
                ctx.parameters[key] = param
        return

    for key in ctx.parameters.keys():
        if key in params:
            ctx.parameters[key] = params[key]
