import argparse
import os

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
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--sweep", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    parser.add_argument("--use-us", default="0", type=str, help="Whether to use TPUs from the USA")
    parser.add_argument("--cleanup", default="0", type=str,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args.sweep, bool(int(args.use_us)), args.cleanup, args.prefix


def main():
    sweep, use_us, cleanup, base_prefix = parse_args()
    for zone, tpu_version, tpu_count, preemptible in CONFIGS:
        us_tpu = zone.startswith('us')
        if us_tpu and not use_us:
            continue
        prefix = zone
        if preemptible:
            prefix += "-preemptible"
        os.system(f'screen -dmS "{prefix}" python3 launch_multiple_runs.py --tpus {tpu_count} --zone {zone}'
                  f' --tpu-version {tpu_version} --data-path gs://ggpt4{"us" * us_tpu}/the-big-char-pile/ '
                  f'--prefix {base_prefix}-{prefix} --preemptible {preemptible} --sweep {sweep} --cleanup {cleanup} '
                  f'--timeout-multiplier {len(CONFIGS)}')


if __name__ == '__main__':
    main()
