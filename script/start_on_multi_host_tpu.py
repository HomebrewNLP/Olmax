import argparse
import inspect
import netrc
import os
import pathlib
import subprocess
import threading
import typing

from launch_multiple_runs import all_tpus


def parse_args() -> typing.Tuple[str, str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", type=str, help="Name of the TPU to use")
    parser.add_argument("--zone", type=str, help="Where the TPU is")
    parser.add_argument("--branch", type=str, help="Git branch to use")
    args = parser.parse_args()
    return args.tpu, args.zone, args.branch


def install(zone: str, name: str, worker: int):
    base = ["gcloud", "alpha", "compute", "tpus", "tpu-vm"]
    args = ["--zone", zone, "--worker", str(worker)]
    name = f"ubuntu@{name}"

    if subprocess.call(base + ["scp", "exec.sh", f"{name}:~/exec.sh"] + args):
        return
    file_path = os.path.abspath(inspect.getfile(inspect.currentframe()))
    if subprocess.call(base + ["scp", str(pathlib.Path(file_path).parent.parent / "config.yaml"),
                               f"{name}:~/config.yaml"] + args):
        return
    if subprocess.call(base + ["ssh", name, "--command", "bash exec.sh"] + args):
        return


def main():
    name, zone, branch = parse_args()
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")
    tpu = [tpu for tpu in all_tpus(zone) if tpu['name'].split('/')[-1] == name][0]
    hosts = len(tpu['networkEndpoints'])
    with open("exec.sh", "w") as f:
        f.write("git clone https://github.com/HomebrewNLP/HomebrewNLP-Jax/ ; "
                "cd HomebrewNLP-Jax ; "
                "mv ../config.yaml config.yaml ; "
                "git fetch ; "
                f"git checkout {branch} ; "
                "git pull ; "
                "bash setup.sh ; "
                f"/home/ubuntu/.local/bin/wandb login {wandb_key} ; "
                "screen -dmS model bash -c 'bash run.sh; sleep 100000'")
    for i in range(hosts):
        threading.Thread(target=install, args=(zone, name, i)).start()


if __name__ == '__main__':
    main()
