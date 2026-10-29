# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed vendor-fix advisory proof contracts

"""Reject altered vendor proofs and exception scope using recorded real scanner findings.

The maintained Mojo lock no longer carries a finding that needs an exception: it moved from conda-forge
python 3.15.0 to 3.15.2. The exception path is therefore exercised on the last reviewed record and on the
verbatim Grype match and Syft identity of the hosted audit that accepted it, with the fixed vendor source
read from its immutable CPython commit.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any
import urllib.request

import pytest

from tests.test_tools.test_pixi_audit_cli import LOCK, ROOT
from tools.security_scan.advisory_exceptions import (
    EXCEPTION_FILE,
    conda_lock_decision,
    load_exceptions,
)

RECORDED = ROOT / "tests/fixtures/grype_conda_python_3_15_0_cve_2025_15367.json"
SOURCE = (
    "https://raw.githubusercontent.com/python/cpython/"
    "d220409a13b77de7a6c30abe16ea90db01528edc/Lib/poplib.py"
)
PROOF_SHA256 = "a6ffff188814b56d95b043c31d9e0dfee51ac6d6e92afe74ab070cb6136b076f"
# poplib.py of the v3.13.0 release commit, which predates the vendor fix.
UNFIXED_SOURCE = (
    "https://raw.githubusercontent.com/python/cpython/"
    "60403a5409ff2c3f3b07dd2ca91a7a3e096839c7/Lib/poplib.py"
)
UNFIXED_SHA256 = "015df92cb7e47cde50adaefec620c23f85cc0624bdf5da4000f0c1733dc03f21"
ARTIFACT_SHA256 = "66b6ac5dd5c64ce34a9c610cacdcc25fe43d7daf29611b6c48036a3c51771374"


def reviewed_record(review_by: str, proof_sha256: str = PROOF_SHA256) -> str:
    """Return the last reviewed exception record, due for review on ``review_by``.

    Parameters
    ----------
    review_by : str
        ISO date written into the record; the committed record carried ``2026-11-13``.
    proof_sha256 : str, optional
        Digest the record requires of the installed proof; the reviewed value by default.

    Returns
    -------
    str
        TOML table with that one record, identical to the committed one apart from the date.
    """
    return (
        "[[exception]]\n"
        f'lock = "{LOCK}"\n'
        'package = "python"\n'
        'version = "3.15.0"\n'
        'advisory = "CVE-2025-15367"\n'
        f'artifact_sha256 = "{ARTIFACT_SHA256}"\n'
        'reason = "fixed_code_present"\n'
        'fixed_version = "3.15.0a6"\n'
        'proof_path = "lib/python3.15/poplib.py"\n'
        f'proof_sha256 = "{proof_sha256}"\n'
        f'source = "{SOURCE}"\n'
        f'review_by = "{review_by}"\n'
        'rationale = "Grype applies an unknown version comparator to the Python prerelease fix '
        'boundary; the installed poplib.py must equal the immutable fixed vendor source."\n'
    )


def due_date() -> str:
    """Return a review date two weeks ahead, so the reviewed record stays current."""
    return (datetime.now(timezone.utc).date() + timedelta(days=14)).isoformat()


def recorded_finding() -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the recorded hosted Grype match and the Syft artifact it names.

    Returns
    -------
    tuple of dict
        The verbatim match and the identity fields of its matched artifact.
    """
    data = json.loads(RECORDED.read_text())
    assert data["match"]["artifact"]["id"] == data["artifact"]["id"]
    assert data["artifact"]["metadata"]["sha256"] == ARTIFACT_SHA256
    return data["match"], data["artifact"]


def vendor_source(url: str, digest: str) -> bytes:
    """Read ``poplib.py`` from an immutable CPython commit and check its digest.

    Parameters
    ----------
    url : str
        Fixed HTTPS address of the file at one CPython commit.
    digest : str
        Expected SHA-256 of the file.

    Returns
    -------
    bytes
        The verified file.
    """
    with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
        payload = response.read(65537)
    assert hashlib.sha256(payload).hexdigest() == digest
    return bytes(payload)


@pytest.mark.parametrize(
    "change",
    [
        "none",
        "bytes",
        "missing",
        "symlink",
        "inner_symlink",
        "unfixed",
        "artifact",
        "fix",
        "advisory",
        "unused",
    ],
)
def test_recorded_finding_cannot_outlive_its_exact_vendor_proof(
    tmp_path: Path, change: str
) -> None:
    """Accept the recorded finding only while its exact proof holds, and refuse every invalidation.

    Parameters
    ----------
    tmp_path : Path
        Real independent Git checkout and an installed-prefix stand-in holding the vendor source.
    change : str
        ``none`` keeps the proof intact; every other value invalidates the proof, the package
        identity or the finding scope. ``unfixed`` installs the pre-fix vendor file under a record
        naming that file's digest, so only the comparison with the fixed vendor source refuses it.
    """
    match, artifact = recorded_finding()
    checkout, prefix = tmp_path / "checkout", tmp_path / "prefix"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    lock = checkout / LOCK
    lock.parent.mkdir(parents=True)
    lock.write_bytes((ROOT / LOCK).read_bytes())
    table = checkout / EXCEPTION_FILE
    table.parent.mkdir(parents=True)
    unfixed = change == "unfixed"
    table.write_text(reviewed_record(due_date(), UNFIXED_SHA256 if unfixed else PROOF_SHA256))
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
    proof = prefix / record["proof_path"]
    proof.parent.mkdir(parents=True)
    if unfixed:
        proof.write_bytes(vendor_source(UNFIXED_SOURCE, UNFIXED_SHA256))
    else:
        proof.write_bytes(vendor_source(SOURCE, PROOF_SHA256))
    actual = [copy.deepcopy(match)]
    inventories = {artifact["id"]: copy.deepcopy(artifact)}
    if change == "bytes":
        proof.write_bytes(proof.read_bytes() + b"\n")
    elif change == "missing":
        proof.unlink()
    elif change == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_bytes(proof.read_bytes())
        proof.unlink()
        proof.symlink_to(outside)
    elif change == "inner_symlink":
        inside = proof.with_name("poplib_copy.py")
        inside.write_bytes(proof.read_bytes())
        proof.unlink()
        proof.symlink_to(inside)
    elif change == "artifact":
        inventories[artifact["id"]]["metadata"]["sha256"] = "0" * 64
    elif change == "fix":
        actual[0]["vulnerability"]["fix"]["versions"] = ["3.16.0"]
    elif change == "advisory":
        actual[0]["vulnerability"]["id"] += "-changed"
    elif change == "unused":
        actual.clear()
    decision = conda_lock_decision(checkout, LOCK, actual, inventories, prefix)
    if change == "none":
        assert decision["passed"] is True and decision["blocked"] == decision["unused"] == []
        assert [row["finding"] for row in decision["accepted"]] == [match]
        assert decision["accepted"][0]["evidence"] == {
            "holds": True,
            "source": SOURCE,
            "proof_sha256": PROOF_SHA256,
        }
        return
    assert decision["passed"] is False and decision["accepted"] == []
    assert decision["blocked"] or decision["unused"]


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        ('package = "python"', 'package = "openssl"'),
        ('version = "3.15.0"', 'version = "3.14.0"'),
        ('advisory = "CVE-2025-15367"', 'advisory = "*"'),
        ('proof_path = "lib/python3.15/poplib.py"', 'proof_path = "../poplib.py"'),
        ('reason = "fixed_code_present"', 'reason = "no_fixed_version_in_channel"'),
        ('advisory = "CVE-2025-15367"', 'advisory = "CVE-2025-15366"'),
        (f'artifact_sha256 = "{ARTIFACT_SHA256}"', 'artifact_sha256 = "unavailable"'),
    ],
)
def test_exception_scope_and_review_cannot_be_broadened(original: str, replacement: str) -> None:
    """Refuse expired, unrelated, wildcarded and unfixed table candidates.

    Parameters
    ----------
    original : str
        Exact field of the reviewed record.
    replacement : str
        Candidate that removes a required acceptance boundary.
    """
    raw = reviewed_record(due_date())
    assert original in raw and len(load_exceptions(raw.encode())) == 1
    with pytest.raises(ValueError):
        load_exceptions(raw.replace(original, replacement).encode())


def test_expired_review_is_refused() -> None:
    """A record past its review date is refused even when every other field holds."""
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    with pytest.raises(ValueError, match="renewed review"):
        load_exceptions(reviewed_record(yesterday).encode())


def test_duplicate_and_unavailable_exception_tables_are_refused() -> None:
    """Reject repeated scope and unavailable tables while accepting an empty list."""
    raw = reviewed_record(due_date()).encode()
    assert len(load_exceptions(raw)) == 1
    assert load_exceptions(b"exception = []\n") == ()
    assert isinstance(load_exceptions((ROOT / EXCEPTION_FILE).read_bytes()), tuple)
    for unavailable in (b"", b"exception = {}\n", raw + b"\n" + raw):
        with pytest.raises(ValueError):
            load_exceptions(unavailable)


def test_exception_for_an_unaudited_lock_is_refused(tmp_path: Path) -> None:
    """A record naming a lock outside the committed Pixi inventory is an error, not a pass.

    Parameters
    ----------
    tmp_path : Path
        Real independent Git checkout without the named lock.
    """
    match, artifact = recorded_finding()
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    (checkout / LOCK).parent.mkdir(parents=True)
    (checkout / LOCK).write_bytes((ROOT / LOCK).read_bytes())
    table = checkout / EXCEPTION_FILE
    table.parent.mkdir(parents=True)
    table.write_text(reviewed_record(due_date()).replace(LOCK, "tools/other/pixi.lock"))
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
    with pytest.raises(ValueError, match="unaudited lock"):
        conda_lock_decision(checkout, LOCK, [match], {artifact["id"]: artifact}, tmp_path)
