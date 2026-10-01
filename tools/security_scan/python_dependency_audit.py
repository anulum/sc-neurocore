# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Complete Python profile dependency audit

"""Audit all Python lock profiles and bind report coverage to input digests."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name
from packaging.version import Version

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.security_scan.python_dependency_profiles import (
    CONSTRAINT_INPUT,
    dependency_batches,
    discover_dependency_profiles,
)

RunCommand = Callable[..., subprocess.CompletedProcess[str]]
AUDIT_SCHEMA = "sc-neurocore.python-profile-audit.v1"


def validate_dependency_report(
    payload: object,
    expected: set[tuple[str, str]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Require exactly the requested identities and valid vulnerability records.

    Parameters
    ----------
    payload : object
        Parsed pip-audit JSON from a real scanner invocation.
    expected : set of tuple of str
        Complete canonical name/query-version set for this batch.

    Returns
    -------
    dict
        Audited dependencies keyed by name/version; findings are retained.

    Raises
    ------
    ValueError
        JSON is malformed, dependencies are skipped/missing/duplicated/extra,
        or vulnerability records cannot be interpreted. Exit zero alone is insufficient.
    """
    if not isinstance(payload, dict) or not isinstance(payload.get("dependencies"), list):
        raise ValueError("Invalid dependency audit report.")
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for dep in payload["dependencies"]:
        if not isinstance(dep, dict) or "skip_reason" in dep:
            raise ValueError("Dependency audit contains an unaudited dependency.")
        if not isinstance(dep.get("name"), str) or not isinstance(dep.get("version"), str):
            raise ValueError("Dependency audit identity is invalid.")
        key = (canonicalize_name(dep["name"]), str(Version(dep["version"])))
        vulns = dep.get("vulns")
        if not isinstance(vulns, list):
            raise ValueError("Dependency audit vulnerability list is invalid.")
        for vuln in vulns:
            if not isinstance(vuln, dict) or not isinstance(vuln.get("id"), str) or not vuln["id"]:
                raise ValueError("Dependency audit finding is invalid.")
            fixes = vuln.get("fix_versions")
            if not isinstance(fixes, list) or any(not isinstance(v, str) for v in fixes):
                raise ValueError("Dependency audit fix versions are invalid.")
        if key in found:
            raise ValueError("Dependency audit contains duplicate identities.")
        found[key] = dep
    if set(found) != expected:
        raise ValueError("Dependency audit does not cover the exact requested versions.")
    return found


def run_dependency_audit(
    *,
    repo_root: Path,
    output_dir: Path,
    run_command: RunCommand = subprocess.run,
) -> dict[str, Any]:
    """Execute pip-audit on every lock and emit an aggregate with coverage proof.

    Parameters
    ----------
    repo_root : Path
        Repository whose complete requirements directory is inventoried.
    output_dir : Path
        Packet directory; reports, query inputs and logs go under security/.
    run_command : callable
        Subprocess executor, with bounded calls and no shell invocation.

    Returns
    -------
    dict
        Aggregate dependencies, profiles and explicit coverage_complete/passed.
        Unreadable inputs, tool errors, stale reports and incomplete coverage fail
        closed. All findings remain blocking; no package is installed or upgraded.
    """
    security = output_dir / "security"
    security.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA,
        "dependencies": [],
        "fixes": [],
        "profiles": [],
        "coverage_complete": False,
        "passed": False,
        "errors": [],
    }
    try:
        snapshot = {
            p.relative_to(repo_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo_root / "requirements").glob("*.txt")
        }
        profiles = discover_dependency_profiles(repo_root)
        if any(snapshot.get(p.path) != p.sha256 for p in profiles):
            raise ValueError("Python audit inputs changed during inventory.")
    except (OSError, ValueError, UnicodeError):
        report["errors"].append("Python lock inventory is missing, invalid or inconsistent.")
        profiles = ()
        snapshot = {}
    tool = shutil.which("pip-audit")
    if tool is None:
        tool = str(Path(sys.executable).parent / "pip-audit")
    dependencies: dict[tuple[str, str], dict[str, Any]] = {}
    for profile in profiles:
        directory = security / "python_profiles" / Path(profile.path).stem
        directory.mkdir(parents=True, exist_ok=True)
        results: dict[tuple[str, str], dict[str, Any]] = {}
        batches: list[dict[str, Any]] = []
        for number, batch in enumerate(dependency_batches(profile), 1):
            query = directory / f"requirements-{number}.txt"
            query.write_text(
                "".join(f"{d.name}=={d.audit_version}\n" for d in batch), encoding="utf-8"
            )
            path = directory / f"pip_audit-{number}.json"
            path.unlink(missing_ok=True)
            command = [
                tool,
                "--strict",
                "--no-deps",
                "--disable-pip",
                "--requirement",
                str(query),
                "--format",
                "json",
                "--progress-spinner",
                "off",
                "--timeout",
                "15",
                "--cache-dir",
                str(security / "pip_audit_cache"),
                "--output",
                str(path),
            ]
            rc: int | None = None
            complete = False
            failure_kind: str | None = None
            report_bytes: bytes | None = None
            try:
                process = run_command(
                    command, cwd=repo_root, capture_output=True, text=True, timeout=240, check=False
                )
                rc = process.returncode
                (directory / f"stdout-{number}.log").write_text(process.stdout, encoding="utf-8")
                (directory / f"stderr-{number}.log").write_text(process.stderr, encoding="utf-8")
                report_bytes = path.read_bytes()
                valid = validate_dependency_report(
                    json.loads(report_bytes),
                    {(d.name, d.audit_version) for d in batch},
                )
                has_findings = any(d["vulns"] for d in valid.values())
                complete = rc == 0 or (rc == 1 and has_findings)
                if complete:
                    results.update(valid)
            except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                failure_kind = type(exc).__name__
            batches.append(
                {
                    "command": command,
                    "returncode": rc,
                    "failure_kind": failure_kind,
                    "coverage_complete": complete,
                    "report": path.relative_to(output_dir).as_posix(),
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest()
                    if report_bytes is not None
                    else None,
                    "query_sha256": hashlib.sha256(query.read_bytes()).hexdigest(),
                }
            )
        profile_complete = all(b["coverage_complete"] for b in batches)
        report["profiles"].append(
            {**asdict(profile), "coverage_complete": profile_complete, "batches": batches}
        )
        for dep in profile.dependencies:
            result = results.get((dep.name, dep.audit_version))
            if result is None:
                continue
            key = (dep.name, dep.version)
            if key not in dependencies:
                dependencies[key] = {
                    **result,
                    "name": dep.name,
                    "version": dep.version,
                    "audit_version": dep.audit_version,
                    "profiles": [],
                }
            else:
                findings = {v["id"]: v for v in dependencies[key]["vulns"]}
                findings.update({v["id"]: v for v in result["vulns"]})
                dependencies[key]["vulns"] = [findings[k] for k in sorted(findings)]
            if profile.path not in dependencies[key]["profiles"]:
                dependencies[key]["profiles"].append(profile.path)
    try:
        current = {
            p.relative_to(repo_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (repo_root / "requirements").glob("*.txt")
        }
    except OSError:
        current = {}
    if snapshot != current:
        report["errors"].append("Python audit inputs changed during execution.")
    report["dependencies"] = [dependencies[k] for k in sorted(dependencies)]
    report["constraints"] = {CONSTRAINT_INPUT: snapshot.get(CONSTRAINT_INPUT)}
    report["coverage_complete"] = (
        bool(profiles)
        and not report["errors"]
        and all(p["coverage_complete"] for p in report["profiles"])
    )
    report["passed"] = report["coverage_complete"] and not any(
        d["vulns"] for d in dependencies.values()
    )
    (security / "pip_audit.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    """Run the real all-profile audit CLI and return nonzero on any incomplete audit.

    Parameters
    ----------
    argv : list of str, optional
        CLI arguments; defaults to the process arguments.

    Returns
    -------
    int
        Zero only for complete coverage with no known vulnerabilities. Writes
        reports under the requested output directory and never installs packages.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    report = run_dependency_audit(repo_root=args.repo_root, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
