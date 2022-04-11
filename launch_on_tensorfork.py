import argparse
import base64
import os
import pathlib
import subprocess

import optuna
import yaml

import wandb
from script.launch_multiple_runs import synchronous_deletion
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
    (use_us, dry, cleanup, base_prefix, us_service_account, eu_service_account, storage_tpu_name,
     storage_tpu_zone, percentile) = parse_args()

    if not cleanup:
        with open("sweep.yaml", 'r') as f:
            config = yaml.safe_load(f.read())
        sweep = wandb.sweep(config, entity=WandB.entity, project=WandB.project)
        synchronous_deletion(storage_tpu_name, storage_tpu_name, storage_tpu_zone)
        os.system(f'while ! gcloud alpha compute tpus tpu-vm create {storage_tpu_name} --zone {storage_tpu_zone} '
                  f'--accelerator-type v2-8 --version v2-alpha; do echo; done')
        password = base64.b32encode(os.urandom(16)).decode().lower().strip('=')
        exec_tpu(storage_tpu_name, storage_tpu_zone, '&&'.join(["sudo apt update",
                                                                "sudo apt upgrade -y",
                                                                "sudo apt install -y postgresql postgresql-contrib",
                                                                "sudo systemctl start postgresql",
                                                                "echo 'host  all  all 0.0.0.0/0 md5' | "
                                                                "sudo tee -a /etc/postgresql/12/main/pg_hba.conf",
                                                                "sudo sed -i \"s/\\#listen_addresses = 'localhost'/"
                                                                "listen_addresses = '*'/g\" "
                                                                "/etc/postgresql/12/main/postgresql.conf",
                                                                "sudo -u postgres psql -c "
                                                                f"\"ALTER USER postgres PASSWORD '{password}';\"",
                                                                "sudo systemctl restart postgresql"]))
        storage_description = yaml.safe_load(subprocess.check_output(["gcloud", "alpha", "compute", "tpus", "tpu-vm",
                                                                      "describe", storage_tpu_name, "--zone",
                                                                      storage_tpu_zone]))
        external_ip = storage_description['networkEndpoints'][0]['accessConfig']['externalIp']
        storage = optuna.storages.RDBStorage(url=f"postgresql://postgres:{password}@{external_ip}:5432/postgres",
                                             heartbeat_interval=60, grace_period=300)
        optuna.create_study(storage, direction=optuna.study.StudyDirection.MINIMIZE, study_name=WandB.entity)
    else:
        sweep = ""
        storage = ""
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
               f'screen -dmS "{prefix}" python3 {main_folder}/script/launch_multiple_runs.py --tpus {tpu_count} '
               f'--zone {zone} --tpu-version {tpu_version} '
               f'--data-path gs://homebrewnlp-{"us" if us_tpu else "eu"}/the-char-pile/ '
               f'--prefix {base_prefix}-{prefix} --preemptible {preemptible} '
               f'--sweep {WandB.entity}/{WandB.project}/{sweep} --cleanup {cleanup} '
               f'--timeout-multiplier {len(CONFIGS)} --service-account {service_account} '
               f"--storage '{storage}\'")
        print(cmd)
        if not dry:
            os.system(cmd)


if __name__ == '__main__':
    main()
