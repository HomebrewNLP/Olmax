import argparse
import datetime
import netrc
import time
from contextlib import nullcontext

import wandb
import yaml

from script.launch_multiple_runs import (OLD_DATA_PATH, create_tpu, exec_tpu, send_to_tpu,
                                         synchronous_deletion, tpu_names)


def exec_command(wandb_key: str, data_path: str, branch: str):
    data_path = data_path.replace("/", "\\/")
    # Bottom one doesn't use , on purpose
    return ' && '.join(("sudo apt --fix-missing --fix-broken install -y git python3 python3-pip",
                        "(rm -rf HomebrewNLP-Jax ; pkill -f python3 ; exit 0)",
                        f"git clone --depth 1 --branch {branch} https://github.com/HomebrewNLP/HomebrewNLP-Jax/",
                        "cd HomebrewNLP-Jax", "(bash setup.sh ; exit 0)",
                        f"/home/ubuntu/.local/bin/wandb login {wandb_key}",
                        f'sed -i "s/{OLD_DATA_PATH}/{data_path}/g" src/context.py',
                        f"mv config.yaml HomebrewNLP-Jax/config.yaml"
                        f'screen -dmS model '
                        f'bash -c "cd HomebrewNLP-Jax ; CONFIG=config.yaml bash run.sh"'))


def recreate(host: str, zone: str, tpu_version: int, preemptible: bool, service_account: str, slices: int):
    if host in tpu_names(zone, preempted=True, deleting=True):
        if host not in tpu_names(zone, preempted=False, deleting=False):
            synchronous_deletion("", host, zone)
            create_tpu(host, zone, tpu_version, preemptible, service_account, nullcontext(), slices)
    else:
        create_tpu(host, zone, tpu_version, preemptible, service_account, nullcontext(), slices)


def start_single(host: str, tpu_version: int, zone: str, data_path: str, preemptible: bool,
                 service_account: str, branch: str, slices: int, run_prefix: str, config_path: str):
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")

    with open(config_path, 'r') as f:
        txt = f.read()
    config = yaml.safe_load(txt)
    idx = 0
    start_step = 0
    wandb_api = wandb.Api()
    config["training"]["do_checkpoint"] = True
    base_checkpoint_path = config["training"]["checkpoint_path"]
    while True:
        try:
            config["wandb"]["name"] = f"{run_prefix}-{idx}"
            config["training"]["start_step"] = start_step
            config["training"]["checkpoint_path"] = f"{base_checkpoint_path}-{idx}"

            recreate(host, zone, tpu_version, preemptible, service_account, slices)
            send_to_tpu(zone, host, "config.yaml", yaml.dump(config))
            send_to_tpu(zone, host, "setup.sh", exec_command(wandb_key, data_path, branch))
            exec_tpu(host, zone, "bash setup.sh")

            while host in tpu_names(zone, preempted=False):
                time.sleep(60)

            idx += 1
            run = wandb_api.run(f"{config['wandb']['entity']}/{config['wandb']['project']}/{config['wandb']['name']}")
            start_step = run.summary["_step"]
            start_step -= start_step % config["training"]["checkpoint_interval"]
            config["training"]["checkpoint_load_path"] = config["training"]["checkpoint_path"]

        except KeyboardInterrupt:
            print(f"{host} - {datetime.datetime.now()}: KeyboardInterrupt received. Killing TPU, then self.")
            synchronous_deletion("", host, zone)
            return


def main():
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
    if args.cleanup:
        synchronous_deletion("", args.host, args.zone)
        return

    return start_single(args.host, args.tpu_version, args.zone, args.data_path, args.preemptible, args.service_account,
                        args.branch, args.slices, args.run_prefix, args.config_path)


if __name__ == '__main__':
    main()
