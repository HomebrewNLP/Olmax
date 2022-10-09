import argparse

import grequests

addresses = ["34.91.235.192", "35.204.237.144", "34.90.213.16", "34.90.251.78"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=1.)
    parser.add_argument("--mass", type=float, default=1.)
    parser.add_argument("--k", type=float, default=256)
    parser.add_argument("--max-prob", type=float, default=1)
    parser.add_argument("--power", type=float, default=1)
    parser.add_argument("--scale", type=float, default=0)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=128)
    args = parser.parse_args()
    args = {"prompt": args.prompt, "temperature": args.temperature, "max_probability_mass": args.mass,
            "max_tokens": args.k, "length": args.length, 'seed': args.seed,
            "max_probability_to_filter": args.max_prob,
            "adaptive_filter_power": args.power, "adaptive_filter_scale": args.scale
            }
    outs = grequests.map(grequests.post(f'http://{addr}:62220/completion', json=args) for addr in addresses)
    out = list(outs)[0]
    response = out.json()["completion"]
    print(response)


if __name__ == '__main__':
    main()
