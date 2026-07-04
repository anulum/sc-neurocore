#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li

"""Run the release-tag security sweep and emit a bounded release summary."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

RELEASE_SECURITY_SWEEP_SCHEMA_VERSION = "sc-neurocore.release-security-sweep.v1"
RunLane = Callable[..., dict[str, Any]]
RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def _project_root() -> Path:
    """Return the repository root that owns this release sweep."""
    return Path(__file__).resolve().parents[2]


def _script_root() -> Path:
    """Return the repository root for sibling tool loading."""
    return Path(__file__).resolve().parents[2]


def _load_module(module_name: str, module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_command(
    command: list[str],
    *,
    repo_root: Path,
    run_command: RunCommand,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return run_command(
            command,
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (
            exc.stdout.decode("utf-8", errors="replace")
            if isinstance(exc.stdout, bytes)
            else exc.stdout
        )
        stderr = (
            exc.stderr.decode("utf-8", errors="replace")
            if isinstance(exc.stderr, bytes)
            else exc.stderr
        )
        return subprocess.CompletedProcess(
            command,
            124,
            stdout=stdout or "",
            stderr="\n".join(
                part
                for part in (stderr or "", f"command timed out after {timeout} seconds")
                if part
            ),
        )


def _tail_lines(text: str, *, limit: int = 12) -> list[str]:
    return text.strip().splitlines()[-limit:]


def build_ci_packet(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Build the offline security packet before scanner outputs are added."""
    module = _load_module(
        "ci_security_packet_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "ci_security_packet.py",
    )
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exit_code = int(
            module.main(["--output-dir", str(output_dir), "--fail-on-missing-required"])
        )
    try:
        summary = json.loads(buffer.getvalue())
    except json.JSONDecodeError:
        summary = {"raw_stdout": buffer.getvalue()}
    if not isinstance(summary, dict):
        summary = {"raw_stdout": buffer.getvalue()}
    summary["passed"] = exit_code == 0
    summary["exit_code"] = exit_code
    summary["schema_version"] = str(summary.get("schema_version", "sc-neurocore.ci-packet-run.v1"))
    return cast(dict[str, Any], summary)


def run_lightweight(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run Ruff, Bandit, and actionlint release scanners."""
    module = _load_module(
        "lightweight_scanners_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_lightweight_security_scanners.py",
    )
    return cast(
        dict[str, Any], module.run_lightweight_scanners(repo_root=repo_root, output_dir=output_dir)
    )


def run_python_compliance(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run Python dependency and licence compliance scanners."""
    module = _load_module(
        "python_compliance_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_python_compliance_scanners.py",
    )
    return cast(
        dict[str, Any],
        module.run_python_compliance_scanners(repo_root=repo_root, output_dir=output_dir),
    )


def run_osv(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run OSV-Scanner over committed lockfiles."""
    module = _load_module(
        "osv_scanner_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_osv_scanners.py",
    )
    return cast(dict[str, Any], module.run_osv_scanner(repo_root=repo_root, output_dir=output_dir))


def run_rust(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run Rust advisory and policy scanners."""
    module = _load_module(
        "rust_scanners_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_rust_security_scanners.py",
    )
    return cast(
        dict[str, Any], module.run_rust_scanners(repo_root=repo_root, output_dir=output_dir)
    )


def run_syft(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Generate and validate a Syft CycloneDX SBOM."""
    module = _load_module(
        "syft_scanner_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_syft_cyclonedx_scanners.py",
    )
    return cast(
        dict[str, Any],
        module.run_syft_cyclonedx_scanner(repo_root=repo_root, output_dir=output_dir),
    )


def run_semgrep(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the repository-owned Semgrep release policy."""
    module = _load_module(
        "semgrep_scanner_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_semgrep_scanners.py",
    )
    return cast(
        dict[str, Any], module.run_semgrep_scanner(repo_root=repo_root, output_dir=output_dir)
    )


def run_gitleaks(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the Gitleaks release secret-detection lane."""
    module = _load_module(
        "gitleaks_scanner_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_gitleaks_scanners.py",
    )
    return cast(
        dict[str, Any], module.run_gitleaks_scanner(repo_root=repo_root, output_dir=output_dir)
    )


def run_trivy_fs(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Run the Trivy filesystem vulnerability release lane."""
    module = _load_module(
        "trivy_fs_scanner_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_trivy_fs_scanners.py",
    )
    return cast(
        dict[str, Any], module.run_trivy_fs_scanner(repo_root=repo_root, output_dir=output_dir)
    )


def run_supply_chain_audit(*, repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Audit the generated SBOM and release dependency metadata offline."""
    module = _load_module(
        "supply_chain_audit_for_release_sweep",
        _script_root() / "tools" / "supply_chain_audit.py",
    )
    report = cast(
        dict[str, Any],
        module.audit_supply_chain(
            sbom_path=output_dir / "security" / "sbom.cdx.json",
            pyproject_path=repo_root / "pyproject.toml",
            requirements_path=repo_root / "requirements" / "release.txt",
            strict=True,
        ),
    )
    report["schema_version"] = "sc-neurocore.supply-chain-audit-run.v1"
    _write_json(output_dir / "security" / "supply_chain_audit.json", report)
    return report


def run_hypothesis_fuzz(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run a bounded Hypothesis subset through real pytest collection."""
    security_dir = output_dir / "security"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_fuzz_bitstream_ir.py",
        "tests/test_fuzz_hdl_source_lowering_inputs.py",
        "tests/test_fuzz_nir_import_inputs.py",
        "-q",
    ]
    result = _run_command(command, repo_root=repo_root, run_command=run_command, timeout=240)
    summary = {
        "schema_version": "sc-neurocore.hypothesis-fuzz-subset.v1",
        "passed": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
    }
    _write_json(security_dir / "hypothesis_fuzz_summary.json", summary)
    return summary


def run_cargo_fuzz(
    *,
    repo_root: Path,
    output_dir: Path,
    max_total_time: int,
) -> dict[str, Any]:
    """Run the bounded Rust cargo-fuzz release subset."""
    module = _load_module(
        "cargo_fuzz_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "run_cargo_fuzz_scanners.py",
    )
    return cast(
        dict[str, Any],
        module.run_cargo_fuzz_scanners(
            repo_root=repo_root,
            output_dir=output_dir,
            selected_targets=("all",),
            max_total_time=max_total_time,
        ),
    )


def run_rust_proptest(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Run Rust proptest integration tests for release fuzz evidence."""
    security_dir = output_dir / "security"
    command = [
        "cargo",
        "test",
        "--manifest-path",
        "engine/Cargo.toml",
        "--test",
        "prop_neuron",
        "--test",
        "prop_layer",
        "--test",
        "prop_kuramoto",
    ]
    result = _run_command(command, repo_root=repo_root, run_command=run_command, timeout=600)
    summary = {
        "schema_version": "sc-neurocore.rust-proptest-subset.v1",
        "passed": result.returncode == 0,
        "command": command,
        "returncode": result.returncode,
        "stdout_tail": _tail_lines(result.stdout),
        "stderr_tail": _tail_lines(result.stderr),
    }
    _write_json(security_dir / "rust_proptest_summary.json", summary)
    return summary


def build_artifact_index(*, output_dir: Path) -> dict[str, Any]:
    """Rebuild the release artifact index after scanner artefacts are present."""
    module = _load_module(
        "release_artifact_index_for_release_sweep",
        _script_root() / "tools" / "security_scan" / "release_security_artifact_index.py",
    )
    manifest_path = _project_root() / "security" / "release_artifacts_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    index = cast(dict[str, Any], module.build_artifact_index(manifest_payload, root=output_dir))
    _write_json(output_dir / "release_security_artifact_index.json", index)
    passed = not index.get("missing_required") and not index.get(
        "missing_required_vulnerability_status"
    )
    return {
        "schema_version": "sc-neurocore.release-artifact-index-run.v1",
        "passed": passed,
        "missing_required": index.get("missing_required", []),
        "missing_required_vulnerability_status": index.get(
            "missing_required_vulnerability_status", []
        ),
    }


def _lane_result(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "passed": payload.get("passed") is True,
        "schema_version": payload.get("schema_version"),
    }


def _append_lane(
    lanes: list[dict[str, Any]],
    *,
    name: str,
    runner: RunLane,
    kwargs: dict[str, Any],
) -> None:
    lanes.append(_lane_result(name, runner(**kwargs)))


def run_release_security_sweep(
    *,
    repo_root: Path,
    output_dir: Path,
    include_fuzz: bool = False,
    fuzz_max_total_time: int = 300,
    build_ci_packet: RunLane = build_ci_packet,
    run_lightweight: RunLane = run_lightweight,
    run_python_compliance: RunLane = run_python_compliance,
    run_osv: RunLane = run_osv,
    run_rust: RunLane = run_rust,
    run_syft: RunLane = run_syft,
    run_semgrep: RunLane = run_semgrep,
    run_gitleaks: RunLane = run_gitleaks,
    run_trivy_fs: RunLane = run_trivy_fs,
    run_supply_chain_audit: RunLane = run_supply_chain_audit,
    run_hypothesis_fuzz: RunLane = run_hypothesis_fuzz,
    run_cargo_fuzz: RunLane = run_cargo_fuzz,
    run_rust_proptest: RunLane = run_rust_proptest,
    build_artifact_index: RunLane = build_artifact_index,
) -> dict[str, Any]:
    """Run release security lanes and write the aggregate summary."""
    output_dir.mkdir(parents=True, exist_ok=True)
    lanes: list[dict[str, Any]] = []
    common = {"repo_root": repo_root, "output_dir": output_dir}

    _append_lane(lanes, name="ci-security-packet", runner=build_ci_packet, kwargs=common)
    _append_lane(lanes, name="lightweight-security-scanners", runner=run_lightweight, kwargs=common)
    _append_lane(
        lanes,
        name="python-compliance-scanners",
        runner=run_python_compliance,
        kwargs=common,
    )
    _append_lane(lanes, name="osv-scanner", runner=run_osv, kwargs=common)
    _append_lane(lanes, name="rust-security-scanners", runner=run_rust, kwargs=common)
    _append_lane(lanes, name="syft-cyclonedx-scanner", runner=run_syft, kwargs=common)
    _append_lane(lanes, name="semgrep-scanner", runner=run_semgrep, kwargs=common)
    _append_lane(lanes, name="gitleaks-scanner", runner=run_gitleaks, kwargs=common)
    _append_lane(lanes, name="trivy-fs-scanner", runner=run_trivy_fs, kwargs=common)
    _append_lane(lanes, name="supply-chain-audit", runner=run_supply_chain_audit, kwargs=common)
    _append_lane(lanes, name="hypothesis-fuzz-subset", runner=run_hypothesis_fuzz, kwargs=common)
    _append_lane(lanes, name="rust-proptest-subset", runner=run_rust_proptest, kwargs=common)
    if include_fuzz:
        _append_lane(
            lanes,
            name="cargo-fuzz",
            runner=run_cargo_fuzz,
            kwargs={**common, "max_total_time": fuzz_max_total_time},
        )
    _append_lane(
        lanes,
        name="release-artifact-index",
        runner=build_artifact_index,
        kwargs={"output_dir": output_dir},
    )

    failed_lanes = [lane["name"] for lane in lanes if lane["passed"] is not True]
    summary = {
        "schema_version": RELEASE_SECURITY_SWEEP_SCHEMA_VERSION,
        "passed": not failed_lanes,
        "output_dir": str(output_dir.resolve()),
        "failed_lanes": failed_lanes,
        "lane_count": len(lanes),
        "lanes": lanes,
    }
    _write_json(output_dir / "security" / "release_security_sweep_summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for release security sweeps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=_project_root())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-fuzz", action="store_true")
    parser.add_argument("--fuzz-max-total-time", type=int, default=300)
    return parser


def main(argv: Sequence[str] | None = None, *, runner: RunLane = run_release_security_sweep) -> int:
    """Run the release security sweep command."""
    args = build_parser().parse_args(argv)
    summary = runner(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        include_fuzz=args.include_fuzz,
        fuzz_max_total_time=args.fuzz_max_total_time,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
