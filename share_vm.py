import argparse
import dataclasses
import string
import typing
from netrc import netrc

import namecheap
import requests
import shortuuid
import tpucare
from tpucare import delete_one_tpu, exec_on_tpu, start_single, tpu_ips

tpucare.LOG_LEVEL = 0
_, _, wandb_key = netrc().authenticators("api.wandb.ai")

IP = requests.get("https://ipinfo.io/ip").text


@dataclasses.dataclass
class TPUContext:
    zone: str
    host: str
    ssh_key: str


class Args:
    subdomain_prefix: str
    namecheap_username: str
    namecheap_api_key: str
    domain_name: str
    host: str
    tpu_version: int
    zone: str
    preemptible: bool
    service_account: str
    slices: int
    cleanup: int
    ssh_key: str


def start_fn(ctx: TPUContext, worker: int):
    exec_on_tpu(ctx.host, ctx.zone, f"echo '{ctx.ssh_key}' >> ~/.ssh/authorized_keys", worker)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, help="Name of the TPU")
    parser.add_argument("--subdomain-prefix", type=str, help="like abc to get abc0.example.com and abc7.example.com")
    parser.add_argument("--namecheap-username", type=str, help="Username used for login on namecheap")
    parser.add_argument("--namecheap-api-key", type=str,
                        help="See https://ap.www.namecheap.com/settings/tools/apiaccess/")
    parser.add_argument("--domain-name", type=str, help="example.com, including the .com")
    parser.add_argument("--ssh-key", type=str, help="like `ssh-rsa <random string> <user>@<machine>`")
    parser.add_argument("--tpu-version", type=int, default=3, help="Which TPU version to create (v2-8 or v3-8)")
    parser.add_argument("--zone", type=str, default="europe-west4-a", help="GCP Zone TPUs get created in")
    parser.add_argument("--preemptible", default=1, type=int,
                        help="Whether to create preemptible or non-preemptible TPUs")
    parser.add_argument("--service-account", type=str,
                        help="Service account that controls permissions of TPU (for example, to ensure EU TPUs won't "
                             "use US data)")
    parser.add_argument("--slices", default=1, type=int,
                        help="How many TPU slices each TPU should have (1=>vX-8, 4=>vX-32)")
    parser.add_argument("--cleanup", default=0, type=int,
                        help="Instead of running something new, kill all tpus. 1 or 0 for y/n")
    return parser.parse_args()


def new_id():
    return str(shortuuid.ShortUUID(alphabet=string.digits + string.ascii_lowercase).random(32))


class CreationCallback:
    def __init__(self, args: Args):
        self.args = args
        self.api = namecheap.Api(args.namecheap_username, args.namecheap_api_key, args.namecheap_username, IP,
                                 sandbox=False, debug=False)

    def _update_ips(self, host: str):
        ips = tpu_ips(host, self.args.zone)
        records = self.api.domains_dns_getHosts(self.args.domain_name)

        records.extend([{"RecordType": "A", "HostName": f"{self.args.subdomain_prefix}{i}", "Address": ip,
                         "MXPref": 10, "TTL": 300
                         } for i, ip in enumerate(ips)])
        records = [self.api._elements_names_fix(x) for x in records]  # skipcq: PYL-W0212
        records = list({r["HostName"]: r for r in records}.values())  # deduplicate, and take last element
        self.api.domains_dns_setHosts(self.args.domain_name, records)

    def __call__(self, host: str, ctx: typing.Optional[TPUContext]) -> TPUContext:
        self._update_ips(host)
        return TPUContext(zone=self.args.zone, host=host, ssh_key=self.args.ssh_key)


def main():
    args = parse_args()
    if args.cleanup:
        delete_one_tpu("", args.host, args.zone)

    start_single(args.host, args.tpu_version, args.zone, args.preemptible, args.service_account, args.slices, start_fn,
                 CreationCallback(args))


if __name__ == '__main__':
    main()
