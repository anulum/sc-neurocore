# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _load_tool() -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    tool_path = repo_root / "tools" / "security_scan" / "run_cargo_fuzz_scanners.py"
    spec = importlib.util.spec_from_file_location("run_cargo_fuzz_scanners", tool_path)
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


def test_manifest_cargo_fuzz_command_uses_packet_runner() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["cargo-fuzz-nightly"]["command"] == (
        "python tools/security_scan/run_cargo_fuzz_scanners.py "
        "--output-dir security/ci-security-packet --target all --max-total-time 300"
    )
    assert scanners["cargo-fuzz-nightly"]["pinned_version"] == "cargo-fuzz==0.13.1"


def test_discovers_fuzz_targets_from_cargo_manifest() -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    targets = tool.discover_fuzz_targets(repo_root)

    assert targets == ["bitstream_ops", "ir_parser"]


def test_runner_executes_each_target_with_bounded_time_and_writes_summary(tmp_path: Path) -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert cwd == repo_root
        assert capture_output is True
        assert text is True
        assert check is False
        assert timeout in {320, 1800}
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("bitstream_ops", "ir_parser"),
        max_total_time=40,
        run_command=fake_run,
    )

    assert [call[:5] for call in calls] == [
        ["cargo", "+nightly", "fuzz", "build", "bitstream_ops"],
        ["cargo", "+nightly", "fuzz", "run", "bitstream_ops"],
        ["cargo", "+nightly", "fuzz", "build", "ir_parser"],
        ["cargo", "+nightly", "fuzz", "run", "ir_parser"],
    ]
    assert all("--fuzz-dir" in call for call in calls)
    run_calls = [call for call in calls if call[3] == "run"]
    assert all("-max_total_time=20" in call for call in run_calls)
    assert all("-runs=0" not in call for call in calls)
    assert summary["passed"] is True
    assert summary["target_count"] == 2
    assert (
        json.loads(
            (tmp_path / "packet" / "security" / "cargo_fuzz_summary.json").read_text(
                encoding="utf-8"
            )
        )["passed"]
        is True
    )


def test_runner_rejects_unknown_target(tmp_path: Path) -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    try:
        tool.run_cargo_fuzz_scanners(
            repo_root=repo_root,
            output_dir=tmp_path / "packet",
            selected_targets=("missing",),
            max_total_time=10,
        )
    except ValueError as exc:
        assert "unknown fuzz targets" in str(exc)
    else:
        raise AssertionError("unknown target was accepted")


def test_runner_reports_target_failures_without_dropping_artifacts(tmp_path: Path) -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, timeout, check
        if command[3] == "build":
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="crash")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "run"
    assert report["build_returncode"] == 0
    assert report["stderr_tail"] == ["crash"]


def test_runner_records_timeout_as_target_failure(tmp_path: Path) -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        if command[3] == "build":
            return subprocess.CompletedProcess(command, 0, stdout="built", stderr="")
        raise subprocess.TimeoutExpired(command, timeout, output="partial", stderr="still running")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "run"
    assert report["returncode"] == 124
    assert report["stdout_tail"] == ["partial"]
    assert report["stderr_tail"] == ["still running", "command timed out after 305 seconds"]


def test_runner_records_build_timeout_before_fuzz_execution(tmp_path: Path) -> None:
    tool = _load_tool()
    repo_root = Path(__file__).resolve().parents[2]
    calls: list[list[str]] = []

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        calls.append(command)
        raise subprocess.TimeoutExpired(command, timeout, output="partial", stderr="compiling")

    summary = tool.run_cargo_fuzz_scanners(
        repo_root=repo_root,
        output_dir=tmp_path / "packet",
        selected_targets=("ir_parser",),
        max_total_time=5,
        build_timeout=7,
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["failed_targets"] == ["ir_parser"]
    assert [call[3] for call in calls] == ["build"]
    report = json.loads(
        (tmp_path / "packet" / "security" / "cargo_fuzz_ir_parser.json").read_text(encoding="utf-8")
    )
    assert report["phase"] == "build"
    assert report["build_returncode"] == 124
    assert report["build_stdout_tail"] == ["partial"]
    assert report["build_stderr_tail"] == ["compiling", "command timed out after 7 seconds"]
