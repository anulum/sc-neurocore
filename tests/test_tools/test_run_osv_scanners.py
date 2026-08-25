# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import tomllib
except ImportError:  # pragma: no cover - Python < 3.11 compatibility
    import tomli as tomllib  # type: ignore[no-redef]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    tool_path = _repo_root() / "tools" / "security_scan" / "run_osv_scanners.py"
    spec = importlib.util.spec_from_file_location("run_osv_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    tool_path = _repo_root() / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_osv_command_uses_packet_runner_and_v2_pin() -> None:
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["osv-scanner"]["pinned_version"] == "osv-scanner==2.3.8"
    assert scanners["osv-scanner"]["command"] == (
        "python tools/security_scan/run_osv_scanners.py --output-dir security/ci-security-packet"
    )
    input_paths = {entry["path"] for entry in scanners["osv-scanner"]["inputs"]}
    assert "tools/security_scan/osv-scanner.toml" in input_paths


def test_osv_config_has_no_vulnerability_exceptions() -> None:
    config = tomllib.loads(
        (_repo_root() / "tools" / "security_scan" / "osv-scanner.toml").read_text(encoding="utf-8")
    )

    assert config.get("IgnoredVulns", []) == []


def test_semgrep_overrides_remove_mcp_waivers() -> None:
    """Patched scanner dependencies must replace, not coexist with, MCP waivers."""
    config = tomllib.loads(
        (_repo_root() / "tools" / "security_scan" / "osv-scanner.toml").read_text(encoding="utf-8")
    )
    ignored = {entry["id"]: entry for entry in config.get("IgnoredVulns", [])}
    overrides = (_repo_root() / "requirements" / "semgrep-overrides.txt").read_text(
        encoding="utf-8"
    )

    for ghsa in ("GHSA-vj7q-gjh5-988w", "GHSA-jpw9-pfvf-9f58", "GHSA-hvrp-rf83-w775"):
        assert ghsa not in ignored
    assert "click==8.3.3" in overrides
    assert "mcp==1.28.1" in overrides


def test_semgrep_hash_lock_materializes_security_overrides() -> None:
    """The installed scanner lock must not silently ignore its safe overrides."""
    locked = (_repo_root() / "requirements" / "semgrep.txt").read_text(encoding="utf-8")

    assert "click==8.3.3" in locked
    assert "mcp==1.28.1" in locked
    assert "click==8.1.8" not in locked
    assert "mcp==1.23.3" not in locked


def test_runner_writes_osv_report_and_summary(tmp_path: Path) -> None:
    tool = _load_tool()
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
        del cwd, capture_output, text, timeout, check
        calls.append(command)
        output_path = Path(command[command.index("--output-file") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "example", "ecosystem": "PyPI"},
                                    "vulnerabilities": [],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_osv_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert calls == [
        [
            "osv-scanner",
            "scan",
            "source",
            "--config",
            "tools/security_scan/osv-scanner.toml",
            "--format",
            "json",
            "--all-packages",
            "--output-file",
            f"{tmp_path / 'packet' / 'security' / 'osv_scanner.json'}",
            "--experimental-no-default-plugins",
            "--experimental-plugins",
            "lockfile",
            "--lockfile",
            "Cargo.lock",
            "--lockfile",
            "fuzz/Cargo.lock",
            "--lockfile",
            "crates/tinysc_riscv/Cargo.lock",
            "--lockfile",
            "crates/evo_substrate_core/Cargo.lock",
            "--lockfile",
            "crates/stochastic_doctor_core/Cargo.lock",
            "--lockfile",
            "crates/autonomous_learning/Cargo.lock",
            "--lockfile",
            "crates/core_engine/Cargo.lock",
            "--lockfile",
            "crates/neuro_symbolic/Cargo.lock",
            "--lockfile",
            "src/sc_neurocore/accel/rust/Cargo.lock",
            "--lockfile",
            "studio/frontend/package-lock.json",
        ]
    ]
    assert summary["passed"] is True
    assert summary["attempts"] == 1
    assert summary["package_count"] == 1
    assert summary["vulnerability_count"] == 0
    assert (
        json.loads(
            (tmp_path / "packet" / "security" / "osv_scanner_summary.json").read_text(
                encoding="utf-8"
            )
        )["passed"]
        is True
    )


def test_runner_fails_when_osv_report_is_missing(tmp_path: Path) -> None:
    tool = _load_tool()

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
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_osv_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["validation_errors"] == ["missing OSV report artifact"]


def test_runner_counts_vulnerabilities_as_blocking(tmp_path: Path) -> None:
    tool = _load_tool()

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
        output_path = Path(command[command.index("--output-file") + 1])
        output_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "example", "ecosystem": "PyPI"},
                                    "vulnerabilities": [{"id": "GHSA-test"}],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="")

    summary = tool.run_osv_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["package_count"] == 1
    assert summary["vulnerability_count"] == 1
    assert summary["validation_errors"] == []


def test_runner_retries_transient_osv_resolver_failure(tmp_path: Path) -> None:
    tool = _load_tool()
    calls = 0
    sleeps: list[float] = []

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
        nonlocal calls
        calls += 1
        output_path = Path(command[command.index("--output-file") + 1])
        if calls == 1:
            output_path.write_text(json.dumps({"results": []}), encoding="utf-8")
            return subprocess.CompletedProcess(
                command,
                127,
                stdout="",
                stderr="rpc error: code = Unavailable desc = service unavailable",
            )
        output_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "packages": [
                                {
                                    "package": {"name": "example", "ecosystem": "PyPI"},
                                    "vulnerabilities": [],
                                }
                            ]
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_osv_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
        sleep=sleeps.append,
    )

    assert summary["passed"] is True
    assert summary["attempts"] == 2
    assert summary["package_count"] == 1
    assert sleeps == [5]
