#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Compare two experimental-suite directories on their common routes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline_suite", help="Path to the baseline suite directory")
    parser.add_argument("candidate_suite", help="Path to the candidate suite directory")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of plain text")
    return parser.parse_args()


def _load_summary(suite_dir: Path) -> dict[str, Any]:
    return json.loads((suite_dir / "suite_summary.json").read_text())


def _route_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {route["route_name"]: route for route in summary["routes"]}


def compare_suites(baseline_suite: Path, candidate_suite: Path) -> dict[str, Any]:
    baseline_summary = _load_summary(baseline_suite)
    candidate_summary = _load_summary(candidate_suite)

    baseline_routes = _route_map(baseline_summary)
    candidate_routes = _route_map(candidate_summary)

    common = sorted(set(baseline_routes) & set(candidate_routes))
    baseline_only = sorted(set(baseline_routes) - set(candidate_routes))
    candidate_only = sorted(set(candidate_routes) - set(baseline_routes))

    comparisons: list[dict[str, Any]] = []
    for route_name in common:
        base = baseline_routes[route_name]
        cand = candidate_routes[route_name]
        base_abs = base["max_abs_diff"]
        cand_abs = cand["max_abs_diff"]
        base_rel = base["max_rel_diff"]
        cand_rel = cand["max_rel_diff"]
        comparisons.append(
            {
                "route_name": route_name,
                "baseline_runs": base["runs"],
                "candidate_runs": cand["runs"],
                "baseline_all_passed": base["all_passed"],
                "candidate_all_passed": cand["all_passed"],
                "baseline_max_abs_diff": base_abs,
                "candidate_max_abs_diff": cand_abs,
                "delta_max_abs_diff": None
                if base_abs is None or cand_abs is None
                else cand_abs - base_abs,
                "baseline_max_rel_diff": base_rel,
                "candidate_max_rel_diff": cand_rel,
                "delta_max_rel_diff": None
                if base_rel is None or cand_rel is None
                else cand_rel - base_rel,
            }
        )

    return {
        "baseline_suite": str(baseline_suite),
        "candidate_suite": str(candidate_suite),
        "common_routes": comparisons,
        "baseline_only_routes": baseline_only,
        "candidate_only_routes": candidate_only,
    }


def _format_text(result: dict[str, Any]) -> str:
    lines = [
        f"baseline_suite: {result['baseline_suite']}",
        f"candidate_suite: {result['candidate_suite']}",
        "",
        "Common routes:",
    ]
    for item in result["common_routes"]:
        lines.append(
            f"- {item['route_name']}: "
            f"abs {item['baseline_max_abs_diff']} -> {item['candidate_max_abs_diff']} "
            f"(delta {item['delta_max_abs_diff']}), "
            f"rel {item['baseline_max_rel_diff']} -> {item['candidate_max_rel_diff']} "
            f"(delta {item['delta_max_rel_diff']})"
        )
    lines.append("")
    lines.append(f"baseline_only_routes: {result['baseline_only_routes']}")
    lines.append(f"candidate_only_routes: {result['candidate_only_routes']}")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    result = compare_suites(Path(args.baseline_suite), Path(args.candidate_suite))
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(_format_text(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
