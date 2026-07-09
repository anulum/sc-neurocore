#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLPerf-SC fixture runner tool

"""Run the deterministic low-load MLPerf-SC fixture benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sc_neurocore.benchmarks import MLPerfSCValidationError, run_mlperf_sc_fixture


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--task", default="synthetic_sc_xor", help="Fixture task name")
    parser.add_argument("--model", default="fixture_sc_linear", help="Fixture model name")
    parser.add_argument("--seed", type=int, default=0, help="Deterministic fixture seed")
    parser.add_argument(
        "--bitstream-length",
        type=int,
        default=256,
        help="Positive stochastic bitstream length",
    )
    args = parser.parse_args(argv)

    try:
        result_path = run_mlperf_sc_fixture(
            output_dir=args.output,
            task=args.task,
            model=args.model,
            seed=args.seed,
            bitstream_length=args.bitstream_length,
        )
    except MLPerfSCValidationError as exc:
        print(f"MLPerf-SC fixture invalid: {exc}", file=sys.stderr)
        return 1
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
