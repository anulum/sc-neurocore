#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark regression scanner runner

"""Compare benchmark JSON artefacts and emit a regression report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

BENCHMARK_REGRESSION_SCHEMA_VERSION = "sc-neurocore.benchmark-regression.v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--current", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-regression-pct", type=float, default=5.0)
    return parser


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _numeric_metrics(payload: Any, *, prefix: str = "") -> Iterator[tuple[str, float]]:
    if isinstance(payload, bool):
        return
    if isinstance(payload, int | float):
        yield prefix, float(payload)
        return
    if isinstance(payload, dict):
        for key, value in sorted(payload.items()):
            if not isinstance(key, str):
                continue
            next_prefix = key if not prefix else f"{prefix}.{key}"
            yield from _numeric_metrics(value, prefix=next_prefix)
        return
    if isinstance(payload, list):
        for index, value in enumerate(payload):
            next_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            yield from _numeric_metrics(value, prefix=next_prefix)


def _delta_pct(*, baseline: float, current: float) -> float:
    if baseline == 0.0:
        if current == 0.0:
            return 0.0
        return 100.0
    return ((current - baseline) / abs(baseline)) * 100.0


def run_benchmark_regression_check(
    *,
    baseline: Path,
    current: Path,
    output: Path,
    max_regression_pct: float = 5.0,
) -> dict[str, Any]:
    baseline_payload = _load_json(baseline)
    current_payload = _load_json(current)
    baseline_metrics = dict(_numeric_metrics(baseline_payload))
    current_metrics = dict(_numeric_metrics(current_payload))

    missing_current = sorted(set(baseline_metrics) - set(current_metrics))
    regressions: list[dict[str, Any]] = []
    for path, baseline_value in sorted(baseline_metrics.items()):
        if path not in current_metrics:
            continue
        current_value = current_metrics[path]
        delta_pct = _delta_pct(baseline=baseline_value, current=current_value)
        if delta_pct > max_regression_pct:
            regressions.append(
                {
                    "path": path,
                    "baseline": baseline_value,
                    "current": current_value,
                    "delta_pct": round(delta_pct, 6),
                }
            )

    report = {
        "SPDX-License-Identifier": "AGPL-3.0-or-later",
        "schema_version": BENCHMARK_REGRESSION_SCHEMA_VERSION,
        "baseline": str(baseline),
        "current": str(current),
        "max_regression_pct": max_regression_pct,
        "metric_count": len(baseline_metrics),
        "current_metric_count": len(current_metrics),
        "missing_current_metrics": missing_current,
        "regression_count": len(regressions),
        "regressions": regressions,
        "passed": not missing_current and not regressions,
    }
    _write_json(output, report)
    return report


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_benchmark_regression_check(
        baseline=args.baseline,
        current=args.current,
        output=args.output,
        max_regression_pct=args.max_regression_pct,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
