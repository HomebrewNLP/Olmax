import argparse

import requests

URL = "https://orbscale.com/"


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
    out = requests.post(URL,
                        json={"prompt": args.prompt, "temperature": args.temperature, "max_probability_mass": args.mass,
                              "max_tokens": args.k, "length": args.length, 'seed': args.seed,
                              "max_probability_to_filter": args.max_prob,
                              "adaptive_filter_power": args.power, "adaptive_filter_scale": args.scale
                              })
    response = out.json()["completion"]
    print(response)


if __name__ == '__main__':
    main()
