# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed vendor-fix advisory proof contracts

"""Reject altered vendor proofs and exception scope using actual scanner findings."""

from __future__ import annotations

import copy
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.test_tools.test_pixi_audit_cli import (
    LOCK,
    ROOT,
    NativeCondaPacket,
    native_pixi_audit as native_pixi_audit,
)
from tools.security_scan.advisory_exceptions import (
    EXCEPTION_FILE,
    conda_lock_decision,
    load_exceptions,
)
from tools.security_scan.conda_dependency_reports import syft_lock_inventory


@pytest.mark.parametrize(
    "change", ["bytes", "missing", "symlink", "artifact", "fix", "advisory", "unused"]
)
def test_native_finding_cannot_outlive_its_exact_vendor_proof(
    tmp_path: Path, native_pixi_audit: NativeCondaPacket, change: str
) -> None:
    """Invalidate a copied actual proof without mutating the installed runtime.

    Parameters
    ----------
    tmp_path : Path
        Real independent Git checkout and copied vendor-source file.
    native_pixi_audit : tuple
        Actual Syft and Grype reports from the public committed-lock audit.
    change : str
        Physical proof, package identity or finding scope to invalidate.
    """
    syft, grype, _, prefix, _ = native_pixi_audit
    artifacts = syft_lock_inventory(syft, (ROOT / LOCK).read_bytes(), prefix)
    assert grype["matches"], "The current declared exception needs its actual native finding."
    checkout, copied_prefix = tmp_path / "checkout", tmp_path / "prefix"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    lock = checkout / LOCK
    lock.parent.mkdir(parents=True)
    lock.write_bytes((ROOT / LOCK).read_bytes())
    table = checkout / EXCEPTION_FILE
    table.parent.mkdir(parents=True)
    table.write_bytes((ROOT / EXCEPTION_FILE).read_bytes())
    subprocess.run(
        ["git", "init", "--initial-branch=main", str(checkout)],
        capture_output=True,
        check=True,
        timeout=30,
    )
    subprocess.run(
        ["git", "add", "requirements", LOCK],
        cwd=checkout,
        capture_output=True,
        check=True,
        timeout=30,
    )
    record = load_exceptions(table.read_bytes())[0]
    proof = copied_prefix / record["proof_path"]
    proof.parent.mkdir(parents=True)
    proof.write_bytes((prefix / record["proof_path"]).read_bytes())
    actual = copy.deepcopy(grype["matches"])
    inventories = copy.deepcopy(artifacts)
    if change == "bytes":
        proof.write_bytes(proof.read_bytes() + b"\n")
    elif change == "missing":
        proof.unlink()
    elif change == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(proof.read_bytes())
        proof.unlink()
        proof.symlink_to(outside)
    elif change == "artifact":
        inventories[actual[0]["artifact"]["id"]]["metadata"]["sha256"] = "0" * 64
    elif change == "fix":
        actual[0]["vulnerability"]["fix"]["versions"] = ["3.16.0"]
    elif change == "advisory":
        actual[0]["vulnerability"]["id"] += "-changed"
    elif change == "unused":
        actual.clear()
    decision = conda_lock_decision(checkout, LOCK, actual, inventories, copied_prefix)
    assert decision["passed"] is False and decision["accepted"] == []
    assert decision["blocked"] or decision["unused"]


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('review_by = "2026-11-13"', 'review_by = "2020-01-01"'),
        ('package = "python"', 'package = "openssl"'),
        ('version = "3.15.0"', 'version = "3.14.0"'),
        ('advisory = "CVE-2025-15367"', 'advisory = "*"'),
        ('proof_path = "lib/python3.15/poplib.py"', 'proof_path = "../poplib.py"'),
        ('reason = "fixed_code_present"', 'reason = "no_fixed_version_in_channel"'),
        ('advisory = "CVE-2025-15367"', 'advisory = "CVE-2025-15366"'),
        (
            'artifact_sha256 = "66b6ac5dd5c64ce34a9c610cacdcc25fe43d7daf29611b6c48036a3c51771374"',
            'artifact_sha256 = "unavailable"',
        ),
    ],
)
def test_exception_scope_and_review_cannot_be_broadened(original: str, replacement: str) -> None:
    """Refuse expired, unrelated, wildcarded and unfixed table candidates.

    Parameters
    ----------
    original : str
        Exact field of the maintained reviewed record.
    replacement : str
        Candidate that removes a required acceptance boundary.
    """
    raw = (ROOT / EXCEPTION_FILE).read_text()
    assert original in raw
    with pytest.raises(ValueError):
        load_exceptions(raw.replace(original, replacement).encode())


def test_duplicate_and_unavailable_exception_tables_are_refused() -> None:
    """Reject repeated scope and unavailable tables while accepting an empty list."""
    raw = (ROOT / EXCEPTION_FILE).read_bytes()
    record = raw[raw.index(b"[[exception]]") :]
    assert len(load_exceptions(raw)) == 1
    assert load_exceptions(b"exception = []\n") == ()
    for unavailable in (b"", b"exception = {}\n", raw + b"\n" + record):
        with pytest.raises(ValueError):
            load_exceptions(unavailable)
