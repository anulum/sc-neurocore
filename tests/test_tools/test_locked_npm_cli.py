# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Real npm identity refusal through the audit CLI

"""Reject omitted npm identities using real Git custody and maintained lock bytes."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "condition",
    [
        "missing-version",
        "version-range",
        "invalid-name",
        "linked",
        "nonregistry",
        "missing-origin",
        "empty",
    ],
)
def test_public_cli_refuses_unanswered_npm_identities(tmp_path: Path, condition: str) -> None:
    """Refuse an incomplete tracked npm identity before submitting any advisory query.

    Parameters
    ----------
    tmp_path : Path
        Fresh real Git checkout and immutable output packet directory.
    condition : str
        Unsupported identity introduced into a copy of the maintained Studio lock.

    Notes
    -----
    No scanner executable or response is replaced. The real CLI refuses the
    unsupported lock before scanner execution and retains the exact lock digest.
    Original project requirement and frontend lock bytes remain untouched.
    """
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    shutil.copy(ROOT / "studio/frontend/package.json", checkout / "package.json")
    lock = checkout / "package-lock.json"
    payload = json.loads((ROOT / "studio/frontend/package-lock.json").read_text())
    entry = payload["packages"]["node_modules/@es-joy/resolve.exports"]
    if condition == "missing-version":
        del entry["version"]
    elif condition == "version-range":
        entry["version"] = "^1.2.0"
    elif condition == "invalid-name":
        entry["name"] = False
    elif condition == "linked":
        entry["link"] = True
    elif condition == "nonregistry":
        entry["resolved"] = "file:local-package"
    elif condition == "missing-origin":
        del entry["resolved"]
    else:
        payload["packages"] = {"": payload["packages"][""]}
    lock.write_text(json.dumps(payload, indent=2) + "\n")
    original = lock.read_bytes()
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        check=True,
        capture_output=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "add", "requirements", "package.json", "package-lock.json"],
        cwd=checkout,
        check=True,
        capture_output=True,
        timeout=30,
    )
    target = tmp_path / "packet"
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.security_scan.locked_dependency_audit",
            "--repo-root",
            str(checkout),
            "--ecosystem",
            "npm",
            "--output-dir",
            str(target),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )
    assert process.returncode == 1
    report = json.loads((target / "summary.json").read_text())
    assert report["passed"] is False and report["errors"] == []
    assert report["audits"] == [
        {"path": "package-lock.json", "passed": False, "error_type": "ValueError"}
    ]
    assert not (target / "lock-1").exists()
    assert lock.read_bytes() == original
