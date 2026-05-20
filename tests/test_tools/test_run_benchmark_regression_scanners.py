# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scan" / "run_benchmark_regression_scanners.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_regression_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_benchmark_regression_command_uses_packet_runner() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["benchmark-regression"]["command"] == (
        "python tools/security_scan/run_benchmark_regression_scanners.py "
        "--baseline benchmarks/baselines/security_side_channel_benchmark.json "
        "--current security/benchmark-current/security_side_channel_benchmark.json "
        "--output security/benchmark_regression.json --max-regression-pct 5.0"
    )


def test_runner_passes_when_current_matches_baseline(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    output = tmp_path / "security" / "benchmark_regression.json"
    payload = {"bench": {"ns_per_call": 100.0, "nested": {"gap": 0.2}}}
    baseline.write_text(json.dumps(payload), encoding="utf-8")
    current.write_text(json.dumps(payload), encoding="utf-8")

    report = tool.run_benchmark_regression_check(
        baseline=baseline,
        current=current,
        output=output,
        max_regression_pct=5.0,
    )

    assert report["passed"] is True
    assert report["regression_count"] == 0
    assert json.loads(output.read_text(encoding="utf-8"))["passed"] is True


def test_runner_fails_on_numeric_regression(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    output = tmp_path / "security" / "benchmark_regression.json"
    baseline.write_text(json.dumps({"bench": {"ns_per_call": 100.0}}), encoding="utf-8")
    current.write_text(json.dumps({"bench": {"ns_per_call": 112.0}}), encoding="utf-8")

    report = tool.run_benchmark_regression_check(
        baseline=baseline,
        current=current,
        output=output,
        max_regression_pct=5.0,
    )

    assert report["passed"] is False
    assert report["regression_count"] == 1
    assert report["regressions"][0]["path"] == "bench.ns_per_call"
    assert report["regressions"][0]["delta_pct"] == 12.0


def test_runner_reports_missing_current_metric(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    output = tmp_path / "security" / "benchmark_regression.json"
    baseline.write_text(json.dumps({"bench": {"ns_per_call": 100.0}}), encoding="utf-8")
    current.write_text(json.dumps({"bench": {}}), encoding="utf-8")

    report = tool.run_benchmark_regression_check(
        baseline=baseline,
        current=current,
        output=output,
        max_regression_pct=5.0,
    )

    assert report["passed"] is False
    assert report["missing_current_metrics"] == ["bench.ns_per_call"]


def test_cli_returns_nonzero_on_regression(tmp_path: Path) -> None:
    tool = _load_tool()
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    output = tmp_path / "security" / "benchmark_regression.json"
    baseline.write_text(json.dumps({"latency": 10.0}), encoding="utf-8")
    current.write_text(json.dumps({"latency": 20.0}), encoding="utf-8")

    exit_code = tool.main(
        [
            "--baseline",
            str(baseline),
            "--current",
            str(current),
            "--output",
            str(output),
            "--max-regression-pct",
            "5",
        ]
    )

    assert exit_code == 1
