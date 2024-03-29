import argparse
import os
import pathlib

import wandb
import yaml

from src.context import WandB

CONFIGS = [("europe-west4-a", 3, 250, 1),
           ("us-central1-a", 3, 200, 1),
           ("us-central1-c", 3, 15, 1),
           ("us-central1-c", 3, 5, 0),
           ("us-central1-b", 2, 150, 1),
           ("us-central1-c", 2, 150, 1),
           ("us-central1-f", 2, 150, 1),
           ("us-central1-a", 2, 5, 0),
           ("us-central1-f", 2, 25, 0),
           ]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="homebrew", help="Name prefix for TPUs")
    parser.add_argument("--us-service-account", type=str, help="EMail of the service account used for american TPUs")
    parser.add_argument("--eu-service-account", type=str, help="EMail of the service account used for european TPUs")
    parser.add_argument("--use-us", default=0, type=int, help="Whether to use TPUs from the USA")
    parser.add_argument("--dry", default=1, type=int, help="Whether to only show what it'd do rather than doing it.")
    parser.add_argument("--branch", default="main", type=str, help="Branch on github to use")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return (bool(args.use_us), bool(args.dry), args.cleanup, args.prefix, args.us_service_account,
            args.eu_service_account, args.branch)


def main():
    (use_us, dry, cleanup, base_prefix, us_service_account, eu_service_account, branch) = parse_args()

    if not cleanup:
        with open("sweep.yaml", 'r') as f:
            config = yaml.safe_load(f.read())
        sweep = wandb.sweep(config, entity=WandB.entity, project=WandB.project)
    else:
        sweep = ""
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
               f'--data-path gs://homebrewnlp-{"us" if us_tpu else "eu"}/the-token-pile/ '
               f'--prefix {base_prefix}-{prefix} --preemptible {preemptible} '
               f'--sweep {WandB.entity}/{WandB.project}/{sweep} --cleanup {cleanup} '
               f'--timeout-multiplier {len(CONFIGS)} --service-account {service_account} '
               f'--branch {branch}')
        print(cmd)
        if not dry:
            os.system(cmd)  # skipcq: BAN-B605


if __name__ == '__main__':
    main()
