import argparse
import datetime
import multiprocessing
import netrc
import os
import threading
import time
import typing

import google.auth
import googleapiclient.discovery

TIMEOUT_MULTIPLIER = 10

API = googleapiclient.discovery.build('tpu', 'v1')
_, PROJECT = google.auth.default()


def exec_command(wandb_key: str, sweep_id: str, host: str, zone: str, data_path: str):
    data_path = data_path.replace("/", "\\/")
    # Bottom one doesn't use , on purpose
    return ' && '.join((f"sudo apt --fix-missing --fix-broken install -y git python3 python3-pip",
                        f"(rm -rf HomebrewNLP-Jax ; pkill -f python3 ; exit 0)",
                        f"git clone --depth 1 https://github.com/HomebrewNLP/HomebrewNLP-Jax/", f"cd HomebrewNLP-Jax",
                        f"(bash setup.sh ; exit 0)", f"/home/ubuntu/.local/bin/wandb login {wandb_key}",
                        f'sed -i "s/gs:\\/\\/ggpt4\\/the-char-pile\\//{data_path}/g" src/context.py',
                        f'screen -dmS model bash -c "cd HomebrewNLP-Jax ; /home/ubuntu/.local/bin/wandb agent --count 1'
                        f' {sweep_id} ; echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone}" || '
                        f'echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone}'))


def send_commands_to_tpu(wandb_key: str, sweep_id: str, host: str, zone: str, data_path: str):
    command = exec_command(wandb_key, sweep_id, host, zone, data_path)
    with open(f'.{host}.sh', 'w') as f:
        f.write(command)
    os.remove(f'.{host}.sh')
    os.system(f"gcloud alpha compute tpus tpu-vm scp .{host}.sh ubuntu@{host}:~/setup.sh --zone {zone}")


def exec_tpu(host: str, zone: str, command: str):
    print(f"running '{command}' ...", end='')
    start_time = time.time()
    os.system(f"gcloud alpha compute tpus tpu-vm ssh ubuntu@{host} --zone {zone} --command '{command}'")
    print(f"done after {time.time() - start_time:.1f}s")


def tpu_names(zone: str, preempted: bool = True, deleting: bool = False, prefix: str = ''):
    zone = 'projects/' + PROJECT + '/locations/' + zone
    while True:
        try:
            tpus = API.projects().locations().nodes().list(parent=zone).execute().get('nodes', [])
            tpus = [t['name'].split('/')[-1] for t in tpus if
                    (deleting or t['state'] != "DELETING") and (preempted or t['state'] != "PREEMPTED")]
            return [t for t in tpus if t.startswith(prefix)]
        except:
            pass


def delete_one_tpu(prefix: str, host: str, zone: str):
    if prefix not in host:
        return
    print(f"\x1b[32;1m  DELETING {host}\x1b[0m")
    os.system(f"echo y | gcloud alpha compute tpus tpu-vm delete {host} --zone {zone} --async")


def delete_all(prefix: str, zone: str):
    while tpu_names(zone, prefix=prefix):
        threads = [threading.Thread(target=delete_one_tpu, args=(prefix, host, zone), daemon=True) for host in
                   tpu_names(zone)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()


def create_tpu(host: str, zone: str, tpu_version: int, tpus: int, preemptible: bool):
    os.system(f'while ! gcloud alpha compute tpus tpu-vm create {host} '
              f'--zone {zone} --accelerator-type v{tpu_version}-8 --version v2-alpha {"--preemptible" * preemptible}; '
              f'do sleep {tpus * TIMEOUT_MULTIPLIER}; done')


def synchronous_deletion(prefix: str, host: str, zone: str):
    delete_one_tpu(prefix, host, zone)
    while host in tpu_names(zone, deleting=True):
        time.sleep(2)


def start_single(prefix: str, tpu_id: int, tpus: int, sweep_id: str, wandb_key: str, tpu_version: int, zone: str,
                 data_path: str, preemptible: bool):
    host = f"{prefix}-{tpu_id}"
    time.sleep((tpu_id - 1) * TIMEOUT_MULTIPLIER)
    if host in tpu_names(zone, preempted=True, deleting=True):
        if host not in tpu_names(zone, preempted=False, deleting=False):
            synchronous_deletion(prefix, host, zone)
            create_tpu(host, zone, tpu_version, tpus, preemptible)
    else:
        create_tpu(host, zone, tpu_version, tpus, preemptible)

    while True:
        try:
            send_commands_to_tpu(wandb_key, sweep_id, host, zone, data_path)
            exec_tpu(host, zone, "bash setup.sh")

            while host in tpu_names(zone, preempted=False):
                time.sleep(5)

            synchronous_deletion(prefix, host, zone)
            create_tpu(host, zone, tpu_version, tpus, preemptible)

        except KeyboardInterrupt:
            print(f"{host} - {datetime.datetime.now()}: KeyboardInterrupt received. Killing TPU, then self.")
            delete_one_tpu(prefix, host, zone)
            return


def start_multiple(prefix: str, tpus: int, sweep_id: str, tpu_version: int, zone: str, data_path: str,
                   preemptible: bool):
    _, _, wandb_key = netrc.netrc().authenticators("api.wandb.ai")
    procs = []
    for tpu_id in range(tpus):
        proc = multiprocessing.Process(target=start_single, daemon=True, args=(
            prefix, tpu_id + 1, tpus, sweep_id, wandb_key, tpu_version, zone, data_path, preemptible))
        proc.start()
        procs.append(proc)
    while all(t.is_alive() for t in procs):
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            print(f"MAIN - {datetime.datetime.now()}: KeyboardInterrupt received. Killing All TPUs, then self.")
            delete_all(prefix, zone)
            return


def parse_args() -> typing.Tuple[int, int, str, str, str, str, bool, bool]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpus", type=int, default=1, help="How many TPUs should be launched")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--prefix", type=str, default="homebrewnlp-preemptible-tuning", help="Name prefix for TPUs")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--data-path", type=str, default="gs://ggpt4/the-char-pile/",
                        help="Where the data is stored. Should be changed to a bucket in the correct region")
    parser.add_argument("--sweep", type=str, help="ID of the Weights and Biases sweep that'll be resumed")
    parser.add_argument("--cleanup", default="0", type=str,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    parser.add_argument("--preemptible", default="1", type=str,
                        help="Whether to create preemptible or non-preemptible TPUs")
    args = parser.parse_args()
    return args.tpus, args.tpu_version, args.prefix, args.zone, args.sweep, args.data_path, bool(
        int(args.cleanup)), bool(int(args.preemptible))


def main():
    tpus, tpu_version, prefix, zone, sweep_id, data_path, cleanup, preemptible = parse_args()
    if cleanup:
        delete_all(prefix, zone)
    else:
        start_multiple(prefix, tpus, sweep_id, tpu_version, zone, data_path, preemptible)


if __name__ == '__main__':
    main()
