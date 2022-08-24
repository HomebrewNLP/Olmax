import argparse
import dataclasses
import string
import typing
from netrc import netrc

import shortuuid
import tpucare
import yaml
from tpucare import delete_one_tpu, exec_command, exec_on_tpu, send_to_tpu, start_single

import wandb

tpucare.LOG_LEVEL = 0
_, _, wandb_key = netrc().authenticators("api.wandb.ai")


@dataclasses.dataclass
class Context:
    zone: str
    host: str
    config: dict
    branch: str


def start_fn(ctx: Context, worker: int):
    setup = f'(bash setup.sh ; mv ~/config.yaml ~/HomebrewNLP-Jax/config.yaml ; exit 0)'
    send_to_tpu(ctx.host, ctx.zone, "config.yaml", yaml.dump(ctx.config), worker)
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       setup_command=setup, run_command=f"CONFIG=config.yaml bash run.sh", branch=ctx.branch,
                       install_python=False)
    send_to_tpu(ctx.host, ctx.zone, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.host, ctx.zone, "bash setup.sh", worker)


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
    parser.add_argument("--storage-prefix", type=str, help="Storage prefix to use for all runs on WandB")
    parser.add_argument("--config-path", type=str, help="Path to config.yaml")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    parser.add_argument("--run-threshold", default=100, type=int, help="How many of the last runs to scan - at most.")
    parser.add_argument("--merge-runs", default=1, type=int,
                        help="Whether to merge all WandB runs into one logstream or keep one for each host.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.cleanup:
        return delete_one_tpu("", args.host, args.zone)

    with open(args.config_path, 'r') as f:
        txt = f.read()
    config = yaml.safe_load(txt)
    config["training"]["do_checkpoint"] = True
    config["data"]["path"] = args.data_path
    config["dims"]["heads"] = 8 * args.slices
    config["wandb"]["name"] = args.host
    if args.merge_runs:
        config["wandb"]["id"] = str(shortuuid.ShortUUID(alphabet=string.digits + string.ascii_lowercase).random(32))

    checkpoint_path = f'{config["training"]["checkpoint_path"]}-{args.storage_prefix}'
    wandb_api = wandb.Api()

    def creation_callback(host: str, ctx: typing.Optional[Context]) -> Context:
        config["training"]["checkpoint_path"] = checkpoint_path

        if ctx is None:  # first call
            return Context(zone=args.zone, host=host, config=config, branch=args.branch)

        try:
            run = wandb_api.run(f'{config["wandb"]["entity"]}/{config["wandb"]["project"]}/{config["wandb"]["id"]}')
            start_step = int(run.summary["_step"])
        except:  # skipcq: FLK-E722
            start_step = 0
        start_step -= start_step % config["training"]["checkpoint_interval"]
        config["training"]["checkpoint_load_path"] = checkpoint_path if start_step > 0 else ""
        config["training"]["start_step"] = start_step
        return Context(zone=args.zone, host=host, config=config, branch=args.branch)

    start_single(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account, args.slices, start_fn,
                 creation_callback)


if __name__ == '__main__':
    main()
