import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--sweep", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    parser.add_argument("--use-us", default="0", type=str, help="Whether to use TPUs from the USA")
    parser.add_argument("--cleanup", default="0", type=str,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args.sweep, bool(int(args.use_us)), bool(int(args.cleanup)), args.prefix


def main():
    sweep, use_us, cleanup, base_prefix = parse_args()
    for zone, tpu_version, tpu_count, preemptible in [("europe-west4-a", 3, 250, 1),  #
                                                      # ("europe-west4-b", 3, 15, 1),  # missing permissions
                                                      # ("europe-west4-c", 3, 15, 1),  # missing permissions
                                                      ("us-central1-a", 3, 200, 1),  #
                                                      ("us-central1-c", 3, 15, 1),  #
                                                      # ("europe-west4-a", 3, 25, 0),  # actively used, so not occupying
                                                      # ("europe-west4-b", 3, 5, 0),  # missing permissions
                                                      # ("europe-west4-c", 3, 5, 0),  # missing permissions
                                                      ("us-central1-c", 3, 5, 0),  #
                                                      # ("europe-west4-b", 2, 15, 1),  # missing permissions
                                                      # ("europe-west4-c", 2, 15, 1),  # missing permissions
                                                      ("us-central1-b", 2, 150, 1),  #
                                                      ("us-central1-c", 2, 150, 1),  #
                                                      ("us-central1-f", 2, 150, 1),  #
                                                      # ("europe-west4-b", 2, 5, 0),  # missing permissions
                                                      # ("europe-west4-c", 2, 5, 0),  # missing permissions
                                                      ("us-central1-a", 2, 5, 0),  #
                                                      ("us-central1-f", 2, 25, 0),  #
                                                      ]:
        us_tpu = zone.startswith('us')
        if us_tpu and not use_us:
            continue
        prefix = zone
        if preemptible:
            prefix += "-preemptible"
        data_path = f"gs://ggpt4{'us' * us_tpu}/the-big-char-pile/"
        args = f"launch_multiple_runs.py --tpus {tpu_count} --zone {zone} --tpu-version {tpu_version} " \
               f"--data-path {data_path} --prefix {base_prefix}-{prefix} --preemptible {preemptible} --sweep {sweep}"
        if cleanup == 1:
            os.system(f'python3 {args} --cleanup 1')
            continue
        os.system(f'screen -dmS "{prefix}" python3 {args} --cleanup 0')


if __name__ == '__main__':
    main()
