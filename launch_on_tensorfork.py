import argparse
import os
import pathlib
import subprocess

import optuna
import yaml

import wandb
from src.context import WandB

CONFIGS = [("europe-west4-a", 3, 250, 1),  #
           # ("europe-west4-b", 3, 15, 1),  # missing permissions
           # ("europe-west4-c", 3, 15, 1),  # missing permissions
           ("us-central1-a", 3, 200, 1),  # broken region?
           # ("us-central1-c", 3, 15, 1),  # broken region
           # ("europe-west4-a", 3, 25, 0),  # actively used, so not occupying
           # ("europe-west4-b", 3, 5, 0),  # missing permissions
           # ("europe-west4-c", 3, 5, 0),  # missing permissions
           # ("us-central1-c", 3, 5, 0),  # wandb's stopping doesn't work
           # ("europe-west4-b", 2, 15, 1),  # missing permissions
           # ("europe-west4-c", 2, 15, 1),  # missing permissions
           ("us-central1-b", 2, 150, 1),  # broken region?
           ("us-central1-c", 2, 150, 1),  # broken region?
           ("us-central1-f", 2, 150, 1),  #
           # ("europe-west4-b", 2, 5, 0),  # missing permissions
           # ("europe-west4-c", 2, 5, 0),  # missing permissions
           # ("us-central1-a", 2, 5, 0),  # wandb's stopping doesn't work
           # ("us-central1-f", 2, 25, 0),  # wandb's stopping doesn't work
           ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="homebrew", help="Name prefix for TPUs")
    parser.add_argument("--us-service-account", type=str, help="EMail of the service account used for american TPUs")
    parser.add_argument("--storage-tpu-name", type=str, help="Name of the non-preemptible TPU used for redis storage")
    parser.add_argument("--storage-tpu-zone", type=str, help="Zone of the non-preemptible TPU used for redis storage")
    parser.add_argument("--eu-service-account", type=str, help="EMail of the service account used for european TPUs")
    parser.add_argument("--use-us", default=0, type=int, help="Whether to use TPUs from the USA")
    parser.add_argument("--dry", default=1, type=int, help="Whether to only show what it'd do rather than doing it.")
    parser.add_argument("--percentile", default=25, type=float, help="Runs below this percentile are eliminated")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return (bool(args.use_us), bool(args.dry), args.cleanup, args.prefix, args.us_service_account,
            args.eu_service_account, args.storage_tpu_name, args.storage_tpu_zone, args.percentile)


def exec_tpu(host: str, zone: str, command: str):
    subprocess.call(["gcloud", "alpha", "compute", "tpus", "tpu-vm", "ssh", f"ubuntu@{host}",
                     f"--zone", zone, "--command", command])


def main():
    sweep = wandb.sweep({"method": "random", "program": "main.py",
                         "command": ["${env}", "python3", "${program}", "${args}"],
                         "parameters": {"placeholder": {"distribution": "uniform", "min": 0, "max": 1}}},
                        entity=WandB.entity, project=WandB.project)

    (use_us, dry, cleanup, base_prefix, us_service_account, eu_service_account, storage_tpu_name,
     storage_tpu_zone, percentile) = parse_args()
    exec_tpu(storage_tpu_name, storage_tpu_zone, '&&'.join(["sudo apt install -y redis", "redis-cli flushall",
                                                            "sudo sed -i 's/127.0.0.1/0.0.0.0/g' /etc/redis/redis.conf",
                                                            "sudo systemctl restart redis"]))
    storage_description = yaml.safe_load(subprocess.check_output(["gcloud", "alpha", "compute", "tpus", "tpu-vm",
                                                                  "describe", storage_tpu_name, "--zone",
                                                                  storage_tpu_zone]))
    internal_ip = storage_description['networkEndpoints'][0]['ipAddress']
    external_ip = storage_description['networkEndpoints'][0]['accessConfig']['externalIp']
    startup_trials = len(CONFIGS) * 16  # on average, there should be 16 TPUs in parallel per region
    optuna.create_study(f'redis://{external_ip}:6379', optuna.samplers.TPESampler(n_startup_trials=startup_trials),
                        optuna.pruners.PercentilePruner(percentile, n_startup_trials=startup_trials,
                                                        n_warmup_steps=4096),  # ~4x max useful warmup
                        direction=optuna.study.StudyDirection.MINIMIZE, study_name=WandB.entity)

    main_folder = pathlib.Path(os.path.abspath(__file__)).parent
    for zone, tpu_version, tpu_count, preemptible in CONFIGS:
        us_tpu = zone.startswith('us')
        if us_tpu and not use_us:
            continue
        service_account = us_service_account if us_tpu else eu_service_account
        prefix = zone.split('-')
        prefix = prefix[0][:2] + prefix[1][0] + prefix[1][-1] + prefix[2][-1]  # us-central1-f -> usc1f
        if preemptible:
            prefix += "-preemptible"

        cmd = (f'export PYTHONPATH="{main_folder}:$PYTHONPATH" && '
               f'screen -dmS "{prefix}" python3 {main_folder}/scripts/launch_multiple_runs.py --tpus {tpu_count} '
               f'--zone {zone} --tpu-version {tpu_version} '
               f'--data-path gs://homebrewnlp-{"us" if us_tpu else "eu"}/the-char-pile/ '
               f'--prefix {base_prefix}-{prefix} --preemptible {preemptible} --sweep {sweep} --cleanup {cleanup} '
               f'--timeout-multiplier {len(CONFIGS)} --service-account {service_account} --storage {internal_ip}')
        print(cmd)
        if not dry:
            os.system(cmd)


if __name__ == '__main__':
    main()
