#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Online O(1) adaptation benchmark tool

"""Generate a deterministic Online O(1) adaptation benchmark JSON report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sc_neurocore.benchmarks import write_online_o1_adaptation_benchmark
from sc_neurocore.learning.online_o1 import OnlineO1Config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON path")
    parser.add_argument("--n-synapses", type=int, default=1024)
    parser.add_argument("--target-weight", type=int, default=192)
    parser.add_argument("--max-pairings", type=int, default=16)
    parser.add_argument("--reward", type=int, default=7)
    parser.add_argument("--weight-bits", type=int, default=8)
    parser.add_argument("--trace-bits", type=int, default=6)
    parser.add_argument("--reward-bits", type=int, default=4)
    parser.add_argument("--learning-shift", type=int, default=3)
    parser.add_argument("--trace-decay-shift", type=int, default=2)
    args = parser.parse_args(argv)

    try:
        write_online_o1_adaptation_benchmark(
            args.output,
            config=OnlineO1Config(
                weight_bits=args.weight_bits,
                trace_bits=args.trace_bits,
                reward_bits=args.reward_bits,
                learning_shift=args.learning_shift,
                trace_decay_shift=args.trace_decay_shift,
            ),
            n_synapses=args.n_synapses,
            target_weight=args.target_weight,
            max_pairings=args.max_pairings,
            reward=args.reward,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"online O(1) adaptation benchmark invalid: {exc}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
