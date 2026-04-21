#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run safe experimental alternative paths and write JSON reports."""

from __future__ import annotations

import argparse
import json
import sys

from sc_neurocore.experimental import (
    AlternativePathConfig,
    AlternativePathMode,
    build_builtin_registry,
)
from sc_neurocore.experimental.builtins import builtin_cases_for_route
from sc_neurocore.experimental.reporting import default_report_path, write_batch_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-routes", action="store_true", help="List built-in routes and exit")
    parser.add_argument("--route", help="Built-in route name to evaluate")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AlternativePathMode],
        default=AlternativePathMode.SHADOW.value,
        help="Execution mode for the route",
    )
    parser.add_argument(
        "--output",
        help="Output JSON path. Defaults to benchmarks/results/experimental_<route>.json",
    )
    parser.add_argument(
        "--absolute-tolerance",
        type=float,
        default=5e-2,
        help="Absolute comparison tolerance",
    )
    parser.add_argument(
        "--relative-tolerance",
        type=float,
        default=1e-1,
        help="Relative comparison tolerance",
    )
    parser.add_argument(
        "--fail-closed",
        action="store_true",
        help="Disable fail-open fallback in candidate mode",
    )
    parser.add_argument(
        "--no-benchmark",
        action="store_true",
        help="Disable timing capture",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Disable output comparison",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    registry = build_builtin_registry()

    if args.list_routes:
        print(json.dumps(registry.describe(), indent=2))
        return 0

    if not args.route:
        print("--route is required unless --list-routes is used", file=sys.stderr)
        return 2

    cases = builtin_cases_for_route(args.route)
    config = AlternativePathConfig(
        enabled=True,
        mode=AlternativePathMode(args.mode),
        fail_open=not args.fail_closed,
        compare_outputs=not args.no_compare,
        benchmark=not args.no_benchmark,
        absolute_tolerance=args.absolute_tolerance,
        relative_tolerance=args.relative_tolerance,
    )
    summary = registry.evaluate(args.route, cases, config)
    output_path = write_batch_report(summary, args.output or default_report_path(args.route))
    print(json.dumps({"output": str(output_path), **summary.to_report()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
