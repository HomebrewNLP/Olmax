import argparse
import dataclasses
import datetime
import netrc
import threading
import time
import typing
from contextlib import nullcontext
from netrc import netrc

import wandb
import yaml
from tpucare import create_tpu, exec_command, exec_on_tpu, send_to_tpu, start_single, synchronous_deletion, tpu_names

_, _, wandb_key = netrc().authenticators("api.wandb.ai")


@dataclasses.dataclass
class Context:
    zone: str
    host: str
    config: dict


def start_fn(ctx: Context, worker: int):
    setup = f'(bash setup.sh ; exit 0)'
    send_to_tpu(ctx.host, ctx.zone, "config.yaml", yaml.dump(ctx.config), worker)
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       setup_command=setup, run_command=f"CONFIG=config.yaml bash run.sh")
    send_to_tpu(ctx.host, ctx.zone, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.host, ctx.zone, "bash setup.sh", worker)


def recreate(host: str, zone: str, tpu_version: int, preemptible: bool, service_account: str, slices: int):
    if host in tpu_names(zone, preempted=True, deleting=True):
        if host not in tpu_names(zone, preempted=False, deleting=False):
            synchronous_deletion("", host, zone)
            create_tpu(host, zone, tpu_version, preemptible, service_account, nullcontext(), slices)
    else:
        create_tpu(host, zone, tpu_version, preemptible, service_account, nullcontext(), slices)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Name of the TPU")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--data-path", type=str, default="gs://ggpt4/the-char-pile/",
                        help="Where the data is stored. Should be changed to a bucket in the correct region")
    parser.add_argument("--preemptible", default=1, type=int,
                        help="Whether to create preemptible or non-preemptible TPUs")
    parser.add_argument("--service-account", type=str,
                        help="Service account that controls permissions of TPU (for example, to ensure EU TPUs won't "
                             "use US data)")
    parser.add_argument("--branch", type=str, default="main", help="Branch on github to use")
    parser.add_argument("--slices", default=1, type=int,
                        help="How many TPU slices each TPU should have (1=>vX-8, 4=>vX-32)")
    parser.add_argument("--run-prefix", type=str, help="Prefix to use for all runs on WandB")
    parser.add_argument("--config-path", type=str, help="Path to config.yaml")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.cleanup:
        return synchronous_deletion("", args.host, args.zone)

    idx = 0

    with open(args.config_path, 'r') as f:
        txt = f.read()
    config = yaml.safe_load(txt)
    config["training"]["do_checkpoint"] = True
    config["data"]["path"] = args.data_path
    base_checkpoint_path = config["training"]["checkpoint_path"]
    wandb_api = wandb.Api()

    def creation_callback(host: str, ctx: typing.Optional[Context]) -> Context:
        nonlocal idx
        idx += 1
        config["wandb"]["name"] = f"{args.host}-{idx}"
        config["training"]["checkpoint_path"] = f"{base_checkpoint_path}-{idx}"

        if ctx is None:
            return Context(zone=args.zone, host=host, config=config)

        start_step = 0
        for run in wandb_api.runs(f"{config['wandb']['entity']}/{config['wandb']['project']}"):
            if run.name == config['wandb']['name']:
                start_step = run.summary["_step"]
                break
        start_step -= start_step % config["training"]["checkpoint_interval"]
        config["training"]["start_step"] = start_step
        config["training"]["checkpoint_load_path"] = f"{base_checkpoint_path}-{idx - 1}"

        return Context(zone=args.zone, host=host, config=config)

    start_single(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account, args.slices, start_fn,
                 creation_callback)


if __name__ == '__main__':
    main()
