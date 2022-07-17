import argparse
import dataclasses
import typing
from netrc import netrc

from tpucare import delete_all, exec_command, exec_on_tpu, send_to_tpu, start_multiple

from src.context import DataContext

_, _, wandb_key = netrc().authenticators("api.wandb.ai")
OLD_DATA_PATH = DataContext.path.replace("/", "\\/")[:-1]  # remove * at the end


@dataclasses.dataclass
class Context:
    zone: str
    host: str
    sweep_id: str
    data_path: str


def start_fn(ctx: Context, worker: int):
    setup = f'(bash setup.sh ; sed -i "s/{OLD_DATA_PATH}/{ctx.data_path}/g" src/context.py; exit 0)'
    cmd = exec_command(repository="https://github.com/HomebrewNLP/HomebrewNLP-Jax", wandb_key=wandb_key,
                       setup_command=setup, run_command=f"/home/ubuntu/.local/bin/wandb agent {ctx.sweep_id}")
    send_to_tpu(ctx.host, ctx.zone, "setup.sh", cmd, worker)
    exec_on_tpu(ctx.host, ctx.zone, "bash setup.sh", worker)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpus", type=int, default=1, help="How many TPUs should be launched")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--data-path", type=str, default="gs://ggpt4/the-char-pile/",
                        help="Where the data is stored. Should be changed to a bucket in the correct region")
    parser.add_argument("--pretrained-path", type=str, default="",
                        help="Where the pretrained embeddings are stored. Should be changed to a bucket in the correct "
                             "region")
    parser.add_argument("--sweep", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    parser.add_argument("--preemptible", default=1, type=int,
                        help="Whether to create preemptible or non-preemptible TPUs")
    parser.add_argument("--timeout-multiplier", default=1, type=int,
                        help="additional timeout multiplier (for launching many script in parallel)")
    parser.add_argument("--service-account", type=str,
                        help="Service account that controls permissions of TPU (for example, to ensure EU TPUs won't "
                             "use US data)")
    parser.add_argument("--branch", type=str, help="Branch on github to use")
    parser.add_argument("--slices", type=int, help="How many TPU slices each TPU should have (1=>vX-8, 4=>vX-32)")
    parser.add_argument("--config-path", type=str, help="Path to sweep's config.yaml")
    args = parser.parse_args()
    return (args.tpus, args.tpu_version, args.prefix, args.zone, args.sweep, args.data_path, args.pretrained_path,
            bool(args.cleanup), bool(args.preemptible), args.timeout_multiplier, args.service_account, args.branch,
            args.slices, args.config_path)


def main():
    (tpus, tpu_version, prefix, zone, sweep_id, data_path, pretrained_path, cleanup, preemptible, timeout_multiplier,
     service_account, branch, slices, config_path) = parse_args()
    if cleanup:
        return delete_all(prefix, zone)

    def creation_callback(host: str, ctx: typing.Optional[Context]) -> Context:
        if ctx is None:
            return Context(zone=zone, host=host, sweep_id=sweep_id, data_path=data_path)
        return ctx

    start_multiple(prefix, tpu_version, zone, preemptible, service_account, slices, start_fn, creation_callback, tpus)


if __name__ == '__main__':
    main()
