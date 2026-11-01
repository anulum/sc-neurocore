# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Syft CycloneDX scanner regression tests

"""Check the declared Syft version and retained CycloneDX packet contracts."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    """Locate the checkout whose scanner configuration is exercised."""
    return Path(__file__).resolve().parents[2]


def _load_tool() -> Any:
    """Load the production CycloneDX packet runner from this checkout."""
    repo_root = _repo_root()
    tool_path = repo_root / "tools" / "security_scan" / "run_syft_cyclonedx_scanners.py"
    spec = importlib.util.spec_from_file_location("run_syft_cyclonedx_scanners", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_manifest_tool() -> Any:
    """Load the maintained scanner catalogue used by release tooling."""
    repo_root = _repo_root()
    tool_path = repo_root / "tools" / "security_scanner_manifest.py"
    spec = importlib.util.spec_from_file_location("security_scanner_manifest", tool_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_syft_command_uses_packet_runner() -> None:
    """Bind the scanner catalogue to the current Syft pin and packet command."""
    manifest_tool = _load_manifest_tool()
    manifest = manifest_tool.build_scanner_manifest()
    scanners = {scanner["name"]: scanner for scanner in manifest["scanners"]}

    assert scanners["syft-cyclonedx"]["command"] == (
        "python tools/security_scan/run_syft_cyclonedx_scanners.py "
        "--output-dir security/ci-security-packet"
    )
    assert scanners["syft-cyclonedx"]["pinned_version"] == "syft==1.55.0"


def test_runner_writes_and_validates_cyclonedx_sbom(tmp_path: Path) -> None:
    """Retain the inventory and enrich its root with the declared project identity."""
    tool = _load_tool()
    calls: list[list[str]] = []
    project_name, project_version = tool._project_identity(_repo_root())

    def fake_run(
        command: list[str],
        *,
        cwd: Path,
        capture_output: bool,
        text: bool,
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        """Supply the stored CycloneDX response for this process contract."""
        del cwd, capture_output, text, timeout, check
        calls.append(command)
        output_spec = command[command.index("--output") + 1]
        output_path = Path(output_spec.split("=", maxsplit=1)[1])
        output_path.write_text(
            json.dumps(
                {
                    "bomFormat": "CycloneDX",
                    "specVersion": "1.6",
                    "components": [{"name": "sc-neurocore", "type": "library"}],
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert calls == [
        [
            "syft",
            ".",
            "--source-name",
            project_name,
            "--source-version",
            project_version,
            "--exclude",
            "**/.git/**",
            "--exclude",
            "**/.cache/**",
            "--exclude",
            "**/.venv/**",
            "--exclude",
            "**/.venv-*/**",
            "--exclude",
            "**/.pixi/**",
            "--exclude",
            "**/build/**",
            "--exclude",
            "**/node_modules/**",
            "--exclude",
            "**/target/**",
            "--output",
            f"cyclonedx-json={tmp_path / 'packet' / 'security' / 'sbom.cdx.json'}",
        ]
    ]
    assert summary["passed"] is True
    assert summary["component_count"] == 1
    sbom = json.loads(
        (tmp_path / "packet" / "security" / "sbom.cdx.json").read_text(encoding="utf-8")
    )
    assert sbom["metadata"]["component"] == {
        "type": "application",
        "name": project_name,
        "version": project_version,
        "licenses": [{"license": {"id": "AGPL-3.0-or-later"}}],
    }
    assert (
        json.loads(
            (tmp_path / "packet" / "security" / "syft_cyclonedx_summary.json").read_text(
                encoding="utf-8"
            )
        )["passed"]
        is True
    )


def test_runner_fails_when_syft_does_not_write_sbom(tmp_path: Path) -> None:
    """Refuse a successful scanner exit that leaves no inventory artifact."""
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
        """Return without producing the required scanner artifact."""
        del cwd, capture_output, text, timeout, check
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert summary["validation_errors"] == ["missing SBOM artifact"]


def test_runner_fails_on_non_cyclonedx_payload(tmp_path: Path) -> None:
    """Refuse an output format that does not satisfy the CycloneDX contract."""
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
        """Supply a retained report carrying an unsupported inventory format."""
        del cwd, capture_output, text, timeout, check
        output_spec = command[command.index("--output") + 1]
        output_path = Path(output_spec.split("=", maxsplit=1)[1])
        output_path.write_text('{"bomFormat":"other"}', encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    summary = tool.run_syft_cyclonedx_scanner(
        repo_root=_repo_root(),
        output_dir=tmp_path / "packet",
        run_command=fake_run,
    )

    assert summary["passed"] is False
    assert "bomFormat must be CycloneDX" in summary["validation_errors"]
