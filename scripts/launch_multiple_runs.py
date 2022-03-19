import argparse
import netrc
import os
import threading
import time
import typing

from tpunicorn.tpu import list_tpus


def exec_tpu(host: str, command: str):
    os.system(f"gcloud alpha compute tpus tpu-vm ssh ubuntu@{host} --command {command}")


def exec_code(host: str, wandb_key: str, sweep_id: str):  # https://wandb.ai/authorize
    exec_tpu(host, f"sudo apt --fix-missing --fix-broken install -y git python3 python3-pip")
    exec_tpu(host, "git clone --depth 1 https://github.com/HomebrewNLP/HomebrewNLP-Jax/")
    exec_tpu(host, "cd HomebrewNLP-Jax && sudo bash setup.sh")
    exec_tpu(host, f"wandb login {wandb_key}")
    exec_tpu(host, f"screen -S model bash -c 'cd HomebrewNLP-Jax && wandb agent --count 1 {sweep_id} ; "
                   f"python3 -c 'from google.cloud import tpu_v2alpha1; "
                   f"tpu_v2alpha1.TpuClient().delete_node(request=tpu_v2alpha1.DeleteNodeRequest(name={host}))"
                   f".result()'")  # Delete via python client as pu doesn't work and gcloud needs manual confirmation


def tpu_names(zone: str, project: str):
    return [t['name'].split('/')[-1] for t in list_tpus(zone, project)]


def start_single(prefix: str, tpu_id: int, sweep_id: str, wandb_key: str, tpu_version: int, zone: str, project: str):
    tpu_name = f"{prefix}-{tpu_id}"
    while True:
        os.system(f'while ! gcloud alpha compute tpus tpu-vm create {tpu_name} '
                  f'--zone {zone} --accelerator-type v{tpu_version}-8 --version v2-alpha --preemptible; '
                  f'do echo "Trying again.."; done')
        exec_code(tpu_name, wandb_key, sweep_id)
        while tpu_name in tpu_names(zone, project):
            time.sleep(60)


def start_multiple(prefix: str, tpus: int, sweep_id: str, tpu_version: int = 3, zone: str = "europe-west4-a"):
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")
    threads = [threading.Thread(target=start_single, args=(prefix, tpu_id, sweep_id, wandb_key, tpu_version, zone))
               for tpu_id in range(tpus)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def parse_args() -> typing.Tuple[int, int, str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpus", type=int, default=1, help="How many TPUs should be launched")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--sweep-id", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    args = parser.parse_args()
    return args.tpus, args.tpu_version, args.prefix, args.zone, args.sweep_id


def main():
    tpus, tpu_version, prefix, zone, sweep_id = parse_args()
    start_multiple(prefix, tpus, sweep_id, tpu_version, zone)


if __name__ == '__main__':
    main()
