import argparse
import netrc
import subprocess
import threading
import typing

from launch_multiple_runs import all_tpus


def parse_args() -> typing.Tuple[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of the TPU to use")
    parser.add_argument("--zone", type=str, help="Where the TPU is")
    args = parser.parse_args()
    return args.tpu, args.zone


def install(zone: str, name: str, worker: int):
    base = ["gcloud", "alpha", "compute", "tpus", "tpu-vm"]
    args = ["--zone", zone, "--worker", str(worker)]
    name = f"ubuntu@{name}"
    if subprocess.call(base + ["scp", "exec.sh", f"{name}:~/exec.sh"] + args):
        return
    if subprocess.call(base + ["ssh", name, "--command", "bash exec.sh"] + args):
        return


def main():
    name, zone = parse_args()
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")
    tpu = [tpu for tpu in all_tpus(zone) if tpu['name'].split('/')[-1] == name][0]
    hosts = len(tpu['networkEndpoints'])
    with open("exec.sh", "w") as f:
        f.write("git clone https://github.com/HomebrewNLP/HomebrewNLP-Jax/ ; "
                "cd HomebrewNLP-Jax ; "
                "bash setup.sh ; "
                f"/home/ubuntu/.local/bin/wandb login {wandb_key} ; "
                "screen -dmS model bash run.sh")
    for i in range(hosts):
        threading.Thread(target=install, args=(zone, name, i)).start()


if __name__ == '__main__':
    main()
