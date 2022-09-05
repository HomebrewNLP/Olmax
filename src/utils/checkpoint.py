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

UPLOAD_RETRIES = 8


@functools.partial(jax.jit, backend="cpu")
def index_weights(weights, idx):
    cpu_device = jax.devices("cpu")[0]
    return jax.device_put(jax.tree_util.tree_map(lambda i: i[idx], weights), cpu_device)


def write(weights, ckpt_dir):
    file_path = ckpt_dir
    for _ in range(UPLOAD_RETRIES):
        try:
            with smart_open(file_path, "wb") as f:
                np.savez(f, **{str(idx): tensor for idx, tensor in enumerate(weights)})
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
        write(index_weights(flattened, shard), f"{ctx.training.checkpoint_path}/{shard}.npz")


def read_shard(ckpt_dir):
    with smart_open(ckpt_dir, "rb") as f:
        buf = f.read()
    f_io = io.BytesIO(buf)
    deserialized = list(np.load(f_io).items())
    return [tensor for idx, tensor in sorted(deserialized, key=lambda x: int(x[0]))]


def deep_replace(d, value):
    if isinstance(d, dict):
        return {k: deep_replace(v, value) for k, v in d.items()}
    return value


def read_ckpt(ctx: Context, ignore: str = '.*optimizer.*'):
    ignore = re.compile(ignore)

    with smart_open(f"{ctx.training.checkpoint_load_path}/structure.json", "r") as f:
        new_structure = f.read()
    new_structure = json.loads(new_structure)
    new_structure = deep_replace(new_structure, jnp.zeros((1,)))
    _, new_structure = jax.tree_util.tree_flatten(new_structure)

    with multiprocessing.pool.ThreadPool(jax.local_device_count()) as p:
        start = time.time()
        paths = [f"{ctx.training.checkpoint_load_path}/{dev.id}_" for dev in jax.local_devices()]
        shards = list(p.map(read_shard, paths))
        print(f"read from disk/gcs in {time.time() - start:.06}s")

    unsharded = []
    for all_shards in zip(*shards):
        x = np.stack(all_shards)
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        unsharded.append(jnp.asarray(x))
    params = jax.tree_util.tree_unflatten(new_structure, unsharded)

    print("Unknown parameters:  ", [p for p in params.keys() if p not in ctx.parameters and not ignore.match(p)])
    print("Unfilled parameters: ", [p for p in ctx.parameters.keys() if p not in params and not ignore.match(p)])

    if not ctx.parameters:
        for key, param in params.items():
            ctx.parameters[key] = param
        return

    for key in ctx.parameters.keys():
        if key in params:
            ctx.parameters[key] = params[key]
