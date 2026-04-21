#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run all built-in experimental routes repeatedly and validate the reports."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from sc_neurocore.experimental import (
    AlternativePathConfig,
    AlternativePathMode,
    build_builtin_registry,
)
from sc_neurocore.experimental.builtins import builtin_cases_for_route
from sc_neurocore.experimental.reporting import default_report_path, write_batch_report

try:
    from tools.validate_experimental_reports import ValidationResult, validate_report
except ModuleNotFoundError:  # direct script execution with PYTHONPATH=src
    from validate_experimental_reports import ValidationResult, validate_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="How many times to run each built-in route",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in AlternativePathMode],
        default=AlternativePathMode.SHADOW.value,
        help="Execution mode for every route in the suite",
    )
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=0.01,
        help="Promotion-gate absolute-difference cap applied to each report",
    )
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=0.01,
        help="Promotion-gate relative-difference cap applied to each report",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for suite artefacts. Defaults to a timestamped directory under benchmarks/results/experimental_runs/",
    )
    parser.add_argument(
        "--real-only",
        action="store_true",
        help="Exclude demo routes and run only the real bounded evidence routes",
    )
    return parser.parse_args()


def _default_output_dir(mode: str, repetitions: int, real_only: bool) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%dT%H%M%S")
    suffix = "real" if real_only else "all"
    return Path("benchmarks/results/experimental_runs") / f"{stamp}_{mode}_r{repetitions}_{suffix}"


def _report_filename(route_name: str, repetition: int) -> str:
    return f"rep_{repetition:02d}_{default_report_path(route_name).name}"


def _suite_summary(
    output_dir: Path,
    mode: str,
    repetitions: int,
    route_names: tuple[str, ...],
    real_only: bool,
    validation_results: list[ValidationResult],
) -> dict[str, Any]:
    grouped: dict[str, list[ValidationResult]] = {}
    for result in validation_results:
        grouped.setdefault(result.route_name, []).append(result)

    routes: list[dict[str, Any]] = []
    for route_name in sorted(grouped):
        items = grouped[route_name]
        routes.append(
            {
                "route_name": route_name,
                "runs": len(items),
                "all_passed": all(item.ok for item in items),
                "max_abs_diff": max(
                    (item.max_abs_diff for item in items if item.max_abs_diff is not None),
                    default=None,
                ),
                "max_rel_diff": max(
                    (item.max_rel_diff for item in items if item.max_rel_diff is not None),
                    default=None,
                ),
                "reports": [str(item.path) for item in items],
            }
        )

    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "repetitions": repetitions,
        "real_only": real_only,
        "selected_routes": list(route_names),
        "route_count": len(routes),
        "all_passed": all(item.ok for item in validation_results),
        "validation_results": [item.to_dict() for item in validation_results],
        "routes": routes,
    }


def _write_summary_files(output_dir: Path, summary: dict[str, Any]) -> None:
    summary_json = output_dir / "suite_summary.json"
    summary_md = output_dir / "suite_summary.md"
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    lines = [
        "# Experimental Suite Summary",
        "",
        f"- output_dir: `{summary['output_dir']}`",
        f"- mode: `{summary['mode']}`",
        f"- repetitions: `{summary['repetitions']}`",
        f"- real_only: `{summary['real_only']}`",
        f"- route_count: `{summary['route_count']}`",
        f"- all_passed: `{summary['all_passed']}`",
        "",
        "| Route | Runs | All passed | Max abs diff | Max rel diff |",
        "|---|---:|:---:|---:|---:|",
    ]
    for route in summary["routes"]:
        lines.append(
            f"| `{route['route_name']}` | {route['runs']} | "
            f"{'yes' if route['all_passed'] else 'no'} | "
            f"{route['max_abs_diff']} | {route['max_rel_diff']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n")


def _select_route_names(route_names: tuple[str, ...], *, real_only: bool) -> tuple[str, ...]:
    if not real_only:
        return route_names
    return tuple(name for name in route_names if not name.startswith("demo."))


def run_suite(
    *,
    repetitions: int,
    mode: str,
    max_abs_diff: float,
    max_rel_diff: float,
    output_dir: Path,
    real_only: bool = False,
) -> dict[str, Any]:
    registry = build_builtin_registry()
    route_names = _select_route_names(registry.names(), real_only=real_only)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AlternativePathConfig(
        enabled=True,
        mode=AlternativePathMode(mode),
        fail_open=True,
        compare_outputs=True,
        benchmark=True,
        absolute_tolerance=max_abs_diff,
        relative_tolerance=max_rel_diff,
    )

    validation_results: list[ValidationResult] = []
    for repetition in range(1, repetitions + 1):
        for route_name in route_names:
            summary = registry.evaluate(route_name, builtin_cases_for_route(route_name), config)
            report_path = output_dir / _report_filename(route_name, repetition)
            write_batch_report(summary, report_path)
            validation_results.append(
                validate_report(
                    summary.to_report(),
                    path=report_path,
                    max_abs_diff=max_abs_diff,
                    max_rel_diff=max_rel_diff,
                    require_mode=mode,
                )
            )

    summary = _suite_summary(
        output_dir,
        mode,
        repetitions,
        route_names,
        real_only,
        validation_results,
    )
    _write_summary_files(output_dir, summary)
    return summary


def main() -> int:
    args = _parse_args()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else _default_output_dir(args.mode, args.repetitions, args.real_only)
    )
    summary = run_suite(
        repetitions=args.repetitions,
        mode=args.mode,
        max_abs_diff=args.max_abs_diff,
        max_rel_diff=args.max_rel_diff,
        output_dir=output_dir,
        real_only=args.real_only,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
