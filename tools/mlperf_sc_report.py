#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLPerf-SC report aggregation tool

"""Aggregate validated MLPerf-SC result records into one report."""

from __future__ import annotations

import argparse
from json import JSONDecodeError
from pathlib import Path
import sys

from sc_neurocore.benchmarks import MLPerfSCValidationError, aggregate_mlperf_sc_results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="MLPerf-SC result JSON files")
    parser.add_argument("--output", required=True, type=Path, help="Output report JSON path")
    args = parser.parse_args(argv)

    try:
        aggregate_mlperf_sc_results(args.results, output_path=args.output)
    except (JSONDecodeError, MLPerfSCValidationError, OSError) as exc:
        print(f"MLPerf-SC aggregation invalid: {exc}", file=sys.stderr)
        return 1
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
