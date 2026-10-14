# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed locked Conda dependency audit

"""Audit a locked Pixi environment with native Syft and Grype."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from tools.security_scan.advisory_exceptions import conda_lock_decision
from tools.security_scan.conda_dependency_reports import (
    GRYPE_DATABASE_URL,
    conda_lock_artifacts,
    grype_matches,
    syft_lock_inventory,
)
from tools.security_scan.locked_dependency_inventory import pixi_package_count
from tools.security_scan.locked_native_process import execute_command


def _metadata(prefix: Path) -> dict[str, str]:
    """Hash the complete installed Conda metadata cohort without following links."""
    paths = sorted((prefix / "conda-meta").glob("*.json"))
    if not paths or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Installed Conda metadata custody is unavailable.")
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def run_conda_audit(repo_root: Path, relative: str, directory: Path) -> bool:
    """Audit the installed default Linux environment and bind it to its lock.

    Parameters
    ----------
    repo_root : Path
        Checkout containing the committed lock and reviewed advisory exceptions.
    relative : str
        Repository-relative Pixi lock path.
    directory : Path
        Fresh directory for native command packets and the decision.

    Returns
    -------
    bool
        True when the complete installed inventory has no unaccepted findings.

    Raises
    ------
    OSError, ValueError
        Installation, scanner, artifact custody or advisory proof is unavailable.

    Notes
    -----
    Conda advisories use Grype's generated CPE identities and NVD matching.
    Complete package inventory is not proof of complete vulnerability coverage.
    """
    lock = repo_root / relative
    raw = lock.read_bytes()
    pixi_package_count(raw)
    conda_lock_artifacts(raw)
    directory.mkdir(parents=True, exist_ok=False)
    rc, _ = execute_command(
        [
            "pixi",
            "install",
            "--no-config",
            "--locked",
            "--manifest-path",
            str(lock.with_suffix(".toml")),
        ],
        repo_root,
        directory / "install",
        json_output=False,
    )
    if rc:
        raise ValueError("The committed Conda environment could not be installed.")
    rc, info = execute_command(
        [
            "pixi",
            "info",
            "--no-config",
            "--json",
            "--manifest-path",
            str(lock.with_suffix(".toml")),
        ],
        repo_root,
        directory / "environment",
    )
    environments = info.get("environments_info") if isinstance(info, dict) else None
    if (
        rc
        or not isinstance(environments, list)
        or len(environments) != 1
        or (
            not isinstance(environments[0], dict)
            or environments[0].get("name") != "default"
            or not isinstance(environments[0].get("prefix"), str)
        )
    ):
        raise ValueError("Native Pixi environment identity is unavailable.")
    prefix = Path(environments[0]["prefix"])
    if not prefix.is_absolute() or not prefix.is_dir():
        raise ValueError("Native Pixi environment directory is unavailable.")
    before = _metadata(prefix)
    syft_config = directory / "syft-config.json"
    syft_config.write_text(json.dumps({"check-for-app-update": False, "parallelism": 1}) + "\n")
    rc, sbom = execute_command(
        [
            "syft",
            "scan",
            "dir:" + str(prefix),
            "--config",
            str(syft_config),
            "--override-default-catalogers",
            "conda-meta-cataloger",
            "--select-catalogers=-file",
            "--parallelism",
            "1",
            "--output",
            "syft-json",
        ],
        repo_root,
        directory / "syft",
    )
    if rc:
        raise ValueError("Native Syft inventory failed.")
    artifacts = syft_lock_inventory(sbom, raw, prefix)
    cache = Path(os.environ.get("GRYPE_DB_CACHE_DIR", str(directory.parent / "grype-cache")))
    grype_config = directory / "grype-config.json"
    grype_config.write_text(
        json.dumps(
            {
                "check-for-app-update": False,
                "ignore": [],
                "exclude": [],
                "show-suppressed": True,
                "include-matcher-suppressions": True,
                "match-upstream-kernel-headers": True,
                "db": {
                    "cache-dir": str(cache.resolve()),
                    "update-url": GRYPE_DATABASE_URL,
                    "auto-update": True,
                    "validate-by-hash-on-start": True,
                    "validate-age": True,
                    "max-allowed-built-age": "120h",
                    "require-update-check": True,
                    "max-update-check-frequency": "0s",
                },
            }
        )
        + "\n"
    )
    rc, report = execute_command(
        [
            "grype",
            "sbom:" + str(directory / "syft/stdout.json"),
            "--config",
            str(grype_config),
            "--output",
            "json",
        ],
        repo_root,
        directory / "grype",
    )
    if rc:
        raise ValueError("Native Grype audit failed.")
    matches = grype_matches(report, artifacts, prefix)
    decision = conda_lock_decision(repo_root, relative, matches, artifacts, prefix)
    if _metadata(prefix) != before or lock.read_bytes() != raw:
        raise ValueError("Conda environment or lock changed during its audit.")
    decision.update(
        {
            "package_inventory_complete": True,
            "package_count": len(artifacts),
            "advisory_matching": "NVD/CPE",
            "limitations": ["Generated CPE matching can miss or misattribute Conda advisories."],
            "metadata_inputs": before,
        }
    )
    (directory / "summary.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision["passed"] is True
