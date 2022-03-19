import argparse
import netrc
import os
import threading
import time
import typing

from tpunicorn.tpu import list_tpus


def exec_tpu(host: str, zone: str, command: str):
    print(f"running '{command}' ...", end='')
    start_time = time.time()
    os.system(f"gcloud alpha compute tpus tpu-vm ssh ubuntu@{host} --zone {zone} --command '{command}'")
    print(f"done after {time.time() - start_time:.1f}s")


def tpu_names(zone: str):
    return [t['name'].split('/')[-1] for t in list_tpus(zone)]


def delete_all(prefix: str, zone: str):
    for host in tpu_names(zone):
        if prefix not in host:
            continue
        try:
            os.system(f"echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone} --async")
        except KeyboardInterrupt:
            break


def start_single(prefix: str, tpu_id: int, sweep_id: str, wandb_key: str, tpu_version: int, zone: str):
    host = f"{prefix}-{tpu_id}"
    while True:
        try:
            os.system(f'while ! gcloud alpha compute tpus tpu-vm create {host} '
                      f'--zone {zone} --accelerator-type v{tpu_version}-8 --version v2-alpha --preemptible; '
                      f'do echo "Trying again.."; done')

            exec_tpu(host, zone, f"sudo apt --fix-missing --fix-broken install -y git python3 python3-pip")
            exec_tpu(host, zone, "rm -rf HomebrewNLP-Jax ; !pkill -f python3")
            exec_tpu(host, zone, "git clone --depth 1 https://github.com/HomebrewNLP/HomebrewNLP-Jax/")
            exec_tpu(host, zone, "cd HomebrewNLP-Jax && bash setup.sh")
            exec_tpu(host, zone, f"wandb login {wandb_key}")
            exec_tpu(host, zone, f'nohup bash -c "cd HomebrewNLP-Jax && wandb agent --count 1 {sweep_id} ; '
                                 f'echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone}" '
                                 f'&> log.txt 2> error.txt &')

            while host in tpu_names(zone):
                time.sleep(60)
        except KeyboardInterrupt:
            print(f"KeyboardInterrupt registered. Killing TPUs {host}. Don't press it a second time.")
            break
    delete_all(prefix, zone)


def start_multiple(prefix: str, tpus: int, sweep_id: str, tpu_version: int = 3, zone: str = "europe-west4-a"):
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")
    threads = [threading.Thread(target=start_single, args=(prefix, tpu_id + 1, sweep_id, wandb_key, tpu_version, zone),
                                daemon=True) for tpu_id in range(tpus)]
    for t in threads:
        t.start()
    while all(t.is_alive() for t in threads):
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print("KeyboardInterrupt registered. Killing all TPUs. Don't press it a second time.")
            break
    delete_all(prefix, zone)


def parse_args() -> typing.Tuple[int, int, str, str, str, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpus", type=int, default=1, help="How many TPUs should be launched")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--sweep-id", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    parser.add_argument("--cleanup", default="0", type=str,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    args = parser.parse_args()
    return args.tpus, args.tpu_version, args.prefix, args.zone, args.sweep_id, bool(int(args.cleanup))


def main():
    tpus, tpu_version, prefix, zone, sweep_id, cleanup = parse_args()
    if cleanup:
        delete_all(prefix, zone)
    else:
        start_multiple(prefix, tpus, sweep_id, tpu_version, zone)


if __name__ == '__main__':
    main()
