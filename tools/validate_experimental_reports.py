#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Validate experimental-route JSON reports against promotion gates."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    path: Path
    route_name: str
    ok: bool
    reasons: list[str]
    max_abs_diff: float | None
    max_rel_diff: float | None
    matched_cases: int
    total_cases: int
    candidate_failures: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "route_name": self.route_name,
            "ok": self.ok,
            "reasons": self.reasons,
            "max_abs_diff": self.max_abs_diff,
            "max_rel_diff": self.max_rel_diff,
            "matched_cases": self.matched_cases,
            "total_cases": self.total_cases,
            "candidate_failures": self.candidate_failures,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        help="Experimental JSON reports to validate. Defaults to benchmarks/results/experimental_*.json",
    )
    parser.add_argument(
        "--max-abs-diff",
        type=float,
        default=None,
        help="Fail if any case exceeds this absolute-difference cap",
    )
    parser.add_argument(
        "--max-rel-diff",
        type=float,
        default=None,
        help="Fail if any case exceeds this relative-difference cap",
    )
    parser.add_argument(
        "--require-mode",
        choices=("shadow", "candidate", "baseline"),
        default=None,
        help="Require all reports to use this execution mode",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of plain text",
    )
    return parser.parse_args()


def _default_report_paths() -> list[Path]:
    return sorted(Path("benchmarks/results").glob("experimental_*.json"))


def _load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _max_diff(cases: list[dict[str, Any]], key: str) -> float | None:
    values = [
        case["comparison"][key]
        for case in cases
        if case.get("comparison") is not None and case["comparison"].get(key) is not None
    ]
    return max(values) if values else None


def validate_report(
    report: dict[str, Any],
    *,
    path: Path,
    max_abs_diff: float | None = None,
    max_rel_diff: float | None = None,
    require_mode: str | None = None,
) -> ValidationResult:
    reasons: list[str] = []
    route_name = str(report.get("route_name", "<unknown>"))
    candidate_failures = int(report.get("candidate_failures", 0))
    matched_cases = int(report.get("matched_cases", 0))
    total_cases = int(report.get("total_cases", 0))
    mode = report.get("mode")
    cases = list(report.get("cases", []))

    if candidate_failures != 0:
        reasons.append(f"candidate_failures={candidate_failures}")
    if matched_cases != total_cases:
        reasons.append(f"matched_cases={matched_cases}/{total_cases}")
    if require_mode is not None and mode != require_mode:
        reasons.append(f"mode={mode!r}, expected {require_mode!r}")

    max_abs_seen = _max_diff(cases, "max_abs_diff")
    max_rel_seen = _max_diff(cases, "max_rel_diff")

    if max_abs_diff is not None and max_abs_seen is not None and max_abs_seen > max_abs_diff:
        reasons.append(f"max_abs_diff={max_abs_seen} > {max_abs_diff}")
    if max_rel_diff is not None and max_rel_seen is not None and max_rel_seen > max_rel_diff:
        reasons.append(f"max_rel_diff={max_rel_seen} > {max_rel_diff}")

    for case in cases:
        comparison = case.get("comparison")
        if comparison is None:
            reasons.append(f"case {case.get('case_name', '<unknown>')} has no comparison block")
            continue
        if not bool(comparison.get("matched", False)):
            reasons.append(f"case {case.get('case_name', '<unknown>')} unmatched")
        if case.get("candidate_error") is not None:
            reasons.append(f"case {case.get('case_name', '<unknown>')} candidate_error present")

    return ValidationResult(
        path=path,
        route_name=route_name,
        ok=not reasons,
        reasons=reasons,
        max_abs_diff=max_abs_seen,
        max_rel_diff=max_rel_seen,
        matched_cases=matched_cases,
        total_cases=total_cases,
        candidate_failures=candidate_failures,
    )


def _format_text(results: list[ValidationResult]) -> str:
    lines: list[str] = []
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"{status} {result.route_name} [{result.path}]")
        lines.append(
            "  "
            f"matched={result.matched_cases}/{result.total_cases} "
            f"candidate_failures={result.candidate_failures} "
            f"max_abs_diff={result.max_abs_diff} "
            f"max_rel_diff={result.max_rel_diff}"
        )
        for reason in result.reasons:
            lines.append(f"  reason: {reason}")
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    report_paths = (
        [Path(item) for item in args.reports] if args.reports else _default_report_paths()
    )
    if not report_paths:
        print("No experimental reports found", file=sys.stderr)
        return 2

    results = [
        validate_report(
            _load_report(path),
            path=path,
            max_abs_diff=args.max_abs_diff,
            max_rel_diff=args.max_rel_diff,
            require_mode=args.require_mode,
        )
        for path in report_paths
    ]

    if args.json:
        print(json.dumps([result.to_dict() for result in results], indent=2))
    else:
        print(_format_text(results))

    return 0 if all(result.ok for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
