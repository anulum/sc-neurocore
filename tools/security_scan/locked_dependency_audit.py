# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Blocking dependency lock audit entry point

"""Audit committed lock inputs and refuse findings or incomplete responses."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
from typing import Any

import yaml

from tools.security_scan.locked_dependency_inventory import (
    cargo_audit_configuration,
    go_queries,
    inventory_locks,
    julia_queries,
    pixi_package_count,
    require_cargo_audit_configuration,
)
from tools.security_scan.locked_dependency_reports import audit_osv_queries, pixi_report_passed
from tools.security_scan.locked_native_reports import (
    npm_lock_packages,
    npm_report_passed,
    rust_report_passed,
)
from tools.security_scan.python_dependency_audit import run_dependency_audit

ECOSYSTEMS = ("python", "rust", "npm", "go", "julia", "pixi")


def _execute(command: list[str], cwd: Path, directory: Path) -> tuple[int, object]:
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "command.json").write_text(json.dumps(command) + "\n", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=360)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    (directory / "stdout.json").write_text(stdout, encoding="utf-8")
    (directory / "stderr.log").write_text(stderr, encoding="utf-8")
    (directory / "receipt.json").write_text(
        json.dumps(
            {
                "returncode": process.returncode,
                "timed_out": timed_out,
                "process_reaped": process.poll() is not None,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    if timed_out:
        raise ValueError("Dependency audit timed out; its process group was stopped.")
    return process.returncode, json.loads(stdout)


def _go_queries(repo_root: Path, relative: str, directory: Path) -> tuple[dict[str, Any], ...]:
    rc, payload = _execute(["go", "mod", "edit", "-json"], (repo_root / relative).parent, directory)
    checksum = (repo_root / relative).with_name("go.sum")
    recorded = None
    if checksum.exists():
        recorded = checksum.read_text(encoding="utf-8")
    return go_queries(rc, payload, recorded)


def _audit_lock(ecosystem: str, relative: str, repo_root: Path, directory: Path) -> bool:
    lock = repo_root / relative
    if ecosystem == "julia":
        report = audit_osv_queries(julia_queries(lock.read_bytes()), directory)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report["passed"] is True
    if ecosystem == "rust":
        cargo_home = Path(os.environ.get("CARGO_HOME", str(Path.home() / ".cargo")))
        if not cargo_home.is_absolute():
            cargo_home = repo_root / cargo_home
        rust_configuration = cargo_audit_configuration(repo_root, cargo_home)
        command = [
            "cargo",
            "audit",
            "--file",
            str(lock),
            "--format",
            "json",
            "--deny",
            "warnings",
            "--url",
            "https://github.com/RustSec/advisory-db",
        ]
    elif ecosystem == "npm":
        npm_packages = npm_lock_packages(lock.read_bytes())
        command = [
            "npm",
            "audit",
            "--package-lock-only",
            "--include=dev",
            "--include=optional",
            "--include=peer",
            "--registry=https://registry.npmjs.org",
            "--json",
        ]
    elif ecosystem == "go":
        # Query every locked version, including checksum-only versions. Call
        # analysis is neither needed nor allowed to prune advisory findings.
        queries = _go_queries(
            repo_root, relative, directory.with_name(directory.name + "-identities")
        )
        report = audit_osv_queries(queries, directory)
        (directory / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        return report["passed"] is True
    else:
        pixi_packages = pixi_package_count(lock.read_bytes())
        command = [
            "pixi",
            "exec",
            "--spec",
            "pixi-audit==0.1.1=ha35fb5c_30",
            "--channel",
            "https://prefix.dev/prefix-dev/prefix-labs",
            "--channel",
            "https://prefix.dev/conda-forge",
            "pixi-audit",
            "--json",
            str(lock),
        ]
    rc, payload = _execute(command, lock.parent if ecosystem == "npm" else repo_root, directory)
    if ecosystem == "rust":
        (directory / "configuration-input.json").write_text(
            json.dumps(rust_configuration, indent=2) + "\n", encoding="utf-8"
        )
        require_cargo_audit_configuration(repo_root, cargo_home, rust_configuration)
        return rust_report_passed(payload) and rc == 0
    if ecosystem == "npm":
        (directory / "validated-packages.json").write_text(
            json.dumps(npm_packages, indent=2) + "\n", encoding="utf-8"
        )
        return npm_report_passed(payload, len(npm_packages)) and rc == 0
    return pixi_report_passed(payload, pixi_packages) and rc == 0


def run_locked_audit(repo_root: Path, output_dir: Path, ecosystem: str) -> dict[str, Any]:
    """Audit one complete ecosystem and bind all inputs before and after use.

    Parameters
    ----------
    repo_root : Path
        Checkout whose Git index defines the committed lock inventory.
    output_dir : Path
        New packet directory; existing packets cannot be overwritten.
    ecosystem : str
        One maintained ecosystem, selected by the unconditional CI matrix.

    Returns
    -------
    dict
        Input hashes, per-lock decisions and an aggregate blocking result.
        Scanner errors and unanswered packages fail without suppressions.
    """
    output_dir.mkdir(parents=True, exist_ok=False)
    report: dict[str, Any] = {
        "schema_version": "sc-neurocore.lock-audit.v1",
        "ecosystem": ecosystem,
        "passed": False,
        "inputs": [],
        "audits": [],
        "errors": [],
    }
    try:
        if ecosystem not in ECOSYSTEMS:
            raise ValueError("Unknown dependency audit ecosystem.")
        inventory = inventory_locks(repo_root)
        report["inputs"] = [asdict(lock) for lock in inventory]
        selected = [lock for lock in inventory if lock.ecosystem == ecosystem]
        if not selected:
            raise ValueError("The required ecosystem has no committed locks.")
        if ecosystem == "python":
            report["audits"].append(
                run_dependency_audit(repo_root=repo_root, output_dir=output_dir)
            )
        else:
            for number, lock in enumerate(selected, 1):
                row: dict[str, Any] = {"path": lock.path, "passed": False}
                try:
                    row["passed"] = _audit_lock(
                        ecosystem, lock.path, repo_root, output_dir / f"lock-{number}"
                    )
                except (OSError, ValueError, subprocess.SubprocessError, yaml.YAMLError) as exc:
                    row["error_type"] = type(exc).__name__
                report["audits"].append(row)
        if inventory_locks(repo_root) != inventory:
            raise ValueError("Committed dependency inputs changed during the audit.")
        report["passed"] = all(row["passed"] is True for row in report["audits"])
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        report["errors"].append(type(exc).__name__)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the blocking lock audit CLI and return nonzero unless fully clean.

    Parameters
    ----------
    argv : list of str, optional
        Command arguments; defaults to the calling process arguments.

    Returns
    -------
    int
        Zero only after complete current-input coverage with no advisories.
        Does not install, update, suppress or rewrite project dependencies.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ecosystem", choices=ECOSYSTEMS, required=True)
    args = parser.parse_args(argv)
    report = run_locked_audit(args.repo_root.resolve(), args.output_dir, args.ecosystem)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
