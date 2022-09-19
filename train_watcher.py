import argparse
import dataclasses
import string
import typing
from netrc import netrc

import shortuuid
import tpucare
import wandb
import yaml
from tpucare import delete_one_tpu, exec_command, exec_on_tpu, send_to_tpu, start_single

from src.context import Context

tpucare.LOG_LEVEL = 0
_, _, wandb_key = netrc().authenticators("api.wandb.ai")


@dataclasses.dataclass
class TPUContext:
    zone: str
    host: str
    config: dict
    branch: str


class Args:
    host: str
    tpu_version: int
    zone: str
    data_path: str
    preemptible: bool
    service_account: str
    branch: str
    slices: int
    storage_prefix: str
    config_path: str
    cleanup: int
    merge_runs: bool


def start_fn(ctx: TPUContext, worker: int):
    setup = '(bash setup.sh ; mv ~/config.yaml ~/HomebrewNLP-Jax/config.yaml ; exit 0)'
    send_to_tpu(ctx.host, ctx.zone, "config.yaml", yaml.dump(ctx.config), worker)
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       setup_command=setup, run_command="CONFIG=config.yaml bash run.sh", branch=ctx.branch,
                       install_python=False)
    send_to_tpu(ctx.host, ctx.zone, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.host, ctx.zone, "bash setup.sh", worker)


def parse_args() -> Args:
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
    parser.add_argument("--storage-prefix", type=str, help="Storage prefix to use for weights on gcloud bucket")
    parser.add_argument("--config-path", type=str, help="Path to config.yaml")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    parser.add_argument("--merge-runs", default=1, type=int,
                        help="Whether to merge all WandB runs into one logstream or keep one for each host.")
    return parser.parse_args()


def new_id():
    return str(shortuuid.ShortUUID(alphabet=string.digits + string.ascii_lowercase).random(32))


class CreationCallback:
    def __init__(self, args: Args):
        self.args = args
        self.restarts = 0

        with open(args.config_path, 'r') as f:  # skipcq: PTC-W6004
            txt = f.read()
        config = yaml.safe_load(txt)
        cfg = Context(config)
        cfg.training.do_checkpoint = True
        cfg.data.path = args.data_path
        cfg.dims.heads = 8 * args.slices
        cfg.wandb.group = args.host

        if args.merge_runs:
            cfg.wandb.id = new_id()

        cfg.training.checkpoint_path = f'{cfg.training.checkpoint_path}-{args.storage_prefix}'
        self.wandb_api = wandb.Api()
        self.cfg = cfg
        self.last_checkpoint_step = 0

    def _prepare_config(self):  # load checkpoint if exists and avoid overwriting logs at 1000 if already up to 1500
        try:
            run = self.wandb_api.run(f'{self.cfg.wandb.entity}/{self.cfg.wandb.project}/{self.cfg.wandb.id}')
            start_step = int(run.summary["_step"])
        except:  # skipcq: FLK-E722
            return  # no logs yet
        if start_step < self.cfg.training.checkpoint_interval:
            self.cfg.training.checkpoint_load_path = ""
            return  # no checkpoint yet

        self.cfg.training.checkpoint_load_path = self.cfg.training.checkpoint_path
        new_checkpoint_step = start_step - start_step % self.cfg.training.checkpoint_interval
        if new_checkpoint_step < self.cfg.training.checkpoint_interval + self.last_checkpoint_step:
            return  # checkpoint, but no new checkpoint

        self.restarts += 1
        self.last_checkpoint_step = new_checkpoint_step
        self.cfg.wandb.id = new_id()

    def __call__(self, host: str, ctx: typing.Optional[TPUContext]) -> TPUContext:
        if ctx is not None:  # every call after 0th
            self._prepare_config()
        self.cfg.wandb.name = f'{self.args.host}-{self.restarts}'
        print(self.cfg)
        return TPUContext(zone=self.args.zone, host=host, config=self.cfg.config(), branch=self.args.branch)


def main():
    args = parse_args()
    if args.cleanup:
        delete_one_tpu("", args.host, args.zone)

    start_single(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account, args.slices, start_fn,
                 CreationCallback(args))


if __name__ == '__main__':
    main()
