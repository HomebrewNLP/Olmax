"""
Adapted from https://github.com/kingoflolz/mesh-transformer-jax/blob/0a75ca9370576ad9d247facf6cb8e9699300e690
/mesh_transformer/checkpoint.py
"""
import datetime
import functools
import io
import json
import multiprocessing
import re
import subprocess
import time
import typing

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import PyTreeDef
from smart_open import open as smart_open

from src.backend import deep_replace, is_main
from src.context import Context, WhileTrainContext

UPLOAD_RETRIES = 8
WCTX_VALUES = ("scalars", "current_step")
TMP_PATH_ADDON = "_____TEMPORARY"
GSUTIL_PATH = "/opt/google-cloud-sdk/bin/gsutil"


def log(arg: str, verbose: bool):
    if verbose:
        print(datetime.datetime.now(), arg)


def write_shard(weights: typing.Any, idx: int, prefix: str, filename: str, verbose: bool):
    path = f"{prefix}/{jax.process_index()}/{idx}/{filename}"
    shard = jax.device_put(jax.tree_util.tree_map(lambda i: i[idx], weights), jax.devices("cpu")[0])
    log(f"Uploading {len(shard)} objects to {path}", verbose)
    for _ in range(UPLOAD_RETRIES):
        try:
            with smart_open(path, "wb") as f:
                np.savez(f, **{str(idx): tensor for idx, tensor in enumerate(shard)})
            return
        except:  # skipcq: FLK-E722
            log(f"Couldn't save to {path}. Retrying now.", verbose)

    log(f"Saving to {path} failed {UPLOAD_RETRIES} times. Skipping this checkpoint.", True)


def cmd(command: str, check: bool = True):
    return subprocess.run(command.split(' '), check=check)


def move_checkpoint(ctx: Context, new: str):
    cmd(f"{GSUTIL_PATH} -m rm -r {new}", False)  # ignore exit code
    cmd(f"{GSUTIL_PATH} -m cp -r {ctx.training.checkpoint_path} {new}")
    cmd(f"{GSUTIL_PATH} -m rm -r {ctx.training.checkpoint_path}")


def write_checkpoint(ctx: Context, verbose: bool = True):
    flattened, jax_structure = jax.tree_util.tree_flatten(ctx.parameters)
    variance, _ = jax.tree_util.tree_flatten(ctx.parameter_variance)  # same structure

    structure = str(jax_structure)  # like "PyTreeDef({'2': {'a': *}})"
    structure = structure.replace('PyTreeDef', '')[1:-1]  # clean up "types"
    structure = structure.replace(': *', ': null').replace("{'", '{"').replace("':", '":')
    structure = structure.replace("', ", '", ').replace(", '", ', "')  # to valid JSON

    if is_main():
        log(f"Saving structure to {ctx.training.checkpoint_path}/structure.json", verbose)
        for _ in range(UPLOAD_RETRIES):
            try:
                with smart_open(f"{ctx.training.checkpoint_path}/structure.json", "w") as f:  # skipcq: PTC-W6004
                    f.write(structure)
                break
            except:  # skipcq: FLK-E722
                log("Couldn't save structure. Retrying now.", verbose)

    for shard in range(jax.local_device_count()):
        for tree, suffix in ((flattened, "parameters"), (variance, "variance")):
            write_shard(tree, shard, ctx.training.checkpoint_path, f"{suffix}.npz", verbose)


@functools.partial(jax.pmap, axis_name="i")
def _sync(x):
    return jax.lax.psum(x, "i") == jax.device_count()


def sync():
    if not _sync(jnp.ones(jax.local_device_count()))[0]:
        raise ValueError


def write_train_checkpoint(wctx: WhileTrainContext, verbose: bool = True):
    real_path = wctx.ctx.training.checkpoint_path
    wctx.ctx.training.checkpoint_path = real_path + TMP_PATH_ADDON

    write_checkpoint(wctx.ctx, verbose)
    for shard in range(jax.local_device_count()):
        for val in WCTX_VALUES:
            write_shard([getattr(wctx, val)], shard, wctx.ctx.training.checkpoint_path, val + ".npz", verbose)

    sync()  # ensure that all nodes wrote their respective checkpoints before moving them collectively
    if is_main():
        move_checkpoint(wctx.ctx, real_path)
    wctx.ctx.training.checkpoint_path = real_path


def read_shard(checkpoint_dir):
    with smart_open(checkpoint_dir, "rb") as f:
        buf = f.read()
    f_io = io.BytesIO(buf)
    deserialized = list(np.load(f_io).items())
    return [tensor for idx, tensor in sorted(deserialized, key=lambda x: int(x[0]))]


def unshard(shards):
    unsharded = []
    for all_shards in zip(*shards):
        x = np.stack(all_shards)
        if x.dtype == np.dtype('V2'):
            x.dtype = jnp.bfloat16
        unsharded.append(x)  # manual jnp.asarray -> replicated; automatic (via jax.pmap) -> parallel (as before)
    return unsharded


def _read_shards(path: str, structure: PyTreeDef, suffix: str):
    with multiprocessing.pool.ThreadPool(jax.local_device_count()) as p:
        start = time.time()
        paths = [f"{path}/{jax.process_index()}/{shard}/{suffix}.npz" for shard in range(jax.local_device_count())]
        shards = list(p.map(read_shard, paths))
        print(f"Loading {suffix} took {time.time() - start:.2f}s")

    return structure.unflatten(unshard(shards))


def _overwrite(new: dict, old: dict, ignore: re.Pattern):
    if not old:
        print("No entries in old dict. Using new dict.")
        for key, param in new.items():
            old[key] = param
        return

    print("Unknown:  ", [p for p in new.keys() if p not in old and not ignore.match(p)])
    print("Unfilled: ", [p for p in old.keys() if p not in new and not ignore.match(p)])

    for key in old.keys():
        if key in new:
            old[key] = new[key]


def read_checkpoint(ctx: Context, ignore: str = '.*optimizer.*', load_variance: bool = False):
    ignore = re.compile(ignore)

    with smart_open(f"{ctx.training.checkpoint_load_path}/structure.json", "r") as f:
        structure = f.read()
    structure = json.loads(structure)
    py_structure = deep_replace(structure, jnp.zeros((1,)))
    _, structure = jax.tree_util.tree_flatten(py_structure)

    _overwrite(_read_shards(ctx.training.checkpoint_load_path, structure, "parameters"), ctx.parameters, ignore)

    if load_variance:
        py_structure = {k: v for k, v in py_structure.items() if "optimizer" not in k}  # no optimizer for param-lr
        _, structure = jax.tree_util.tree_flatten(py_structure)
        _overwrite(_read_shards(ctx.training.checkpoint_load_path, structure, "variance"), ctx.parameter_variance,
                   ignore)


def read_train_checkpoint(wctx: WhileTrainContext, ignore: str = '.*optimizer.*'):
    _, structure = jax.tree_util.tree_flatten([jnp.zeros((1,))])
    for val in WCTX_VALUES:
        setattr(wctx, val, _read_shards(wctx.ctx.training.checkpoint_load_path, structure, val)[0])
    read_checkpoint(wctx.ctx, ignore, load_variance=True)
