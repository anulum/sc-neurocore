#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Summarise SHD Vertex corrected-selection run artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "sc-neurocore.shd-vertex-summary.v1"


@dataclass(frozen=True)
class RunSummary:
    """Summary of one SHD Vertex run artifact directory."""

    run: str
    seed: int | None
    dcls_version: str | None
    round_each_epoch: bool | None
    sigma_init: float | None
    sigma_final: float | None
    best_fpga_deployable_epoch: int | None
    best_fpga_deployable_test_acc: float | None
    rounding_drop: float | None
    last_epoch: int | None
    last_test_after_round: float | None
    best_native_val_epoch: int | None
    best_native_val_acc: float | None
    best_native_fpga_val_acc: float | None
    best_fpga_val_epoch: int | None
    best_fpga_val_acc: float | None
    best_fpga_native_val_acc: float | None
    test_rows: int
    artifact_dir: str


def summarise_runs(root: Path) -> dict[str, Any]:
    """Return a JSON-serialisable summary for all run directories under root."""
    runs = [
        summarise_run(path)
        for path in sorted(root.iterdir())
        if path.is_dir() and (path / "config.json").exists()
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "root": str(root),
        "run_count": len(runs),
        "runs": [asdict(run) for run in runs],
        "aggregate": _aggregate(runs),
    }


def summarise_run(path: Path) -> RunSummary:
    """Summarise one artifact directory containing config.json and optional log."""
    config = json.loads((path / "config.json").read_text())
    rows = _read_training_log(path / "training_log.csv")
    best_native = _max_row(rows, "val_acc")
    best_fpga = _max_row(rows, "fpga_val_acc")

    return RunSummary(
        run=path.name,
        seed=_optional_int(config.get("seed")),
        dcls_version=_optional_str(config.get("dcls_version")),
        round_each_epoch=_optional_bool(config.get("round_each_epoch")),
        sigma_init=_optional_float(config.get("sigma_init")),
        sigma_final=_optional_float(config.get("sigma_final")),
        best_fpga_deployable_epoch=_optional_int(config.get("best_fpga_deployable_epoch")),
        best_fpga_deployable_test_acc=_optional_float(config.get("fpga_deployable_test_acc")),
        rounding_drop=_optional_float(config.get("rounding_drop")),
        last_epoch=_optional_int(config.get("last_epoch")),
        last_test_after_round=_optional_float(config.get("last_test_at_sig_final_after_round")),
        best_native_val_epoch=_optional_int(best_native.get("epoch")) if best_native else None,
        best_native_val_acc=_optional_float(best_native.get("val_acc")) if best_native else None,
        best_native_fpga_val_acc=_optional_float(best_native.get("fpga_val_acc"))
        if best_native
        else None,
        best_fpga_val_epoch=_optional_int(best_fpga.get("epoch")) if best_fpga else None,
        best_fpga_val_acc=_optional_float(best_fpga.get("fpga_val_acc")) if best_fpga else None,
        best_fpga_native_val_acc=_optional_float(best_fpga.get("val_acc")) if best_fpga else None,
        test_rows=sum(
            1
            for row in rows
            if _optional_float(row.get("test_acc")) is not None and float(row["test_acc"]) >= 0.0
        ),
        artifact_dir=str(path),
    )


def write_outputs(summary: dict[str, Any], out_prefix: Path) -> tuple[Path, Path, Path]:
    """Write JSON, CSV, and Markdown summaries with a shared prefix."""
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_suffix(".json")
    csv_path = out_prefix.with_suffix(".csv")
    md_path = out_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _write_csv(summary["runs"], csv_path)
    md_path.write_text(render_markdown(summary))
    return json_path, csv_path, md_path


def render_markdown(summary: dict[str, Any]) -> str:
    """Render a concise Markdown summary for internal SHD status notes."""
    lines = [
        "# SHD Vertex Corrected-Selection Summary",
        "",
        f"Schema: `{summary['schema_version']}`",
        f"Run count: {summary['run_count']}",
        "",
        "## Aggregate",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in summary["aggregate"].items():
        lines.append(f"| `{key}` | {_format_value(value)} |")

    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| Run | Seed | Round each epoch | Deployable test | Rounding drop | Best FPGA val epoch | Best native val epoch |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for run in summary["runs"]:
        lines.append(
            "| "
            f"`{run['run']}` | {_format_value(run['seed'])} | "
            f"{_format_value(run['round_each_epoch'])} | "
            f"{_format_value(run['best_fpga_deployable_test_acc'])} | "
            f"{_format_value(run['rounding_drop'])} | "
            f"{_format_value(run['best_fpga_val_epoch'])} | "
            f"{_format_value(run['best_native_val_epoch'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def _read_training_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _max_row(rows: Sequence[dict[str, Any]], key: str) -> dict[str, Any] | None:
    usable = [row for row in rows if _optional_float(row.get(key)) is not None]
    if not usable:
        return None
    return max(usable, key=lambda row: float(row[key]))


def _aggregate(runs: Sequence[RunSummary]) -> dict[str, Any]:
    deployable = [
        run.best_fpga_deployable_test_acc
        for run in runs
        if run.best_fpga_deployable_test_acc is not None
    ]
    return {
        "completed_runs": len(runs),
        "deployable_test_mean": _mean(deployable),
        "deployable_test_min": min(deployable) if deployable else None,
        "deployable_test_max": max(deployable) if deployable else None,
        "zero_rounding_drop_runs": sum(1 for run in runs if run.rounding_drop == 0.0),
        "round_each_epoch_runs": sum(1 for run in runs if run.round_each_epoch is True),
    }


def _write_csv(runs: Sequence[dict[str, Any]], path: Path) -> None:
    fieldnames = list(
        asdict(
            RunSummary(
                "",
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                "",
            )
        ).keys()
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(runs)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (str, bytes, bytearray, int, float)):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_value(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _default_out_prefix(root: Path) -> Path:
    safe_name = re.sub(r"[^A-Za-z0-9]+", "_", root.name).strip("_") or "shd_vertex_runs"
    return Path("docs/internal") / f"{safe_name}_summary"


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/masquelier_shd/cloud_results"),
        help="Directory containing SHD run artifact subdirectories",
    )
    parser.add_argument(
        "--out-prefix",
        type=Path,
        default=None,
        help="Output prefix for JSON/CSV/Markdown files",
    )
    parser.add_argument("--json", action="store_true", help="Print summary JSON to stdout")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SHD Vertex summary CLI."""
    args = _parse_args(argv)
    root = args.root.resolve()
    summary = summarise_runs(root)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    out_prefix = args.out_prefix or _default_out_prefix(root)
    if not out_prefix.is_absolute():
        out_prefix = Path.cwd() / out_prefix
    paths = write_outputs(summary, out_prefix)
    for path in paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
