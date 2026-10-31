# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Installed vendor-fix advisory proof contracts

"""Reject altered vendor proofs and exception scope using recorded real scanner findings.

The fixed-code path is exercised on the last reviewed CVE-2025-15367 record and the verbatim Grype match
and Syft identity of the hosted audit that accepted it, with the fixed vendor source read from its
immutable CPython commit. The unreleased-fix path is exercised on the committed CVE-2026-103995 record,
the hosted match that blocked it, the immutable CPython 3.15.2 release file and a recorded conda-forge
version list.
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
    CONDA_FORGE_PYTHON_INDEX,
    EXCEPTION_FILE,
    conda_lock_decision,
    load_exceptions,
    newer_vendor_releases,
    read_url,
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
        ('reason = "fixed_code_present"', 'reason = "vendor_fix_unreleased"'),
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


UNRELEASED_RECORDED = ROOT / "tests/fixtures/grype_conda_python_3_15_2_cve_2026_103995.json"
CHANNEL_VERSIONS = ROOT / "tests/fixtures/conda_forge_python_versions_2026_10_31.json"
UNRELEASED_SOURCE = (
    "https://raw.githubusercontent.com/python/cpython/"
    "5e58eaad7c58df58dfa8ea549c87996e01dd1077/Lib/http/client.py"
)
UNRELEASED_PROOF_SHA256 = "f4506633b5fe4fe06add63d9e32ab5e6e7cda84ec1df365da7faa5c1315fdbc5"
UNRELEASED_ARTIFACT_SHA256 = "dd6fdef748df5636029673a834ca3632ce44c86d71e4986d70bcdb2ae5607a25"


def unreleased_record(review_by: str) -> str:
    """Return the committed unreleased-fix record, due for review on ``review_by``.

    Parameters
    ----------
    review_by : str
        ISO date written into the record.

    Returns
    -------
    str
        TOML table with that one record, identical to the committed one apart from the date.
    """
    committed = (ROOT / EXCEPTION_FILE).read_text()
    start = committed.index("[[exception]]")
    return committed[start:].replace('review_by = "2026-11-14"', f'review_by = "{review_by}"')


def channel_index(*extra: str) -> bytes:
    """Return the recorded conda-forge version list as a package document, with extra versions.

    Parameters
    ----------
    *extra : str
        Versions appended to the recorded list.

    Returns
    -------
    bytes
        JSON document with a ``versions`` member, as the channel API returns it.
    """
    versions = json.loads(CHANNEL_VERSIONS.read_text())["versions"]
    return json.dumps({"versions": [*versions, *extra]}).encode()


def unreleased_checkout(tmp_path: Path, review_by: str) -> tuple[Path, Path, dict[str, str]]:
    """Build a real Git checkout holding the lock and the record, and the installed proof file.

    Parameters
    ----------
    tmp_path : Path
        Disposable root of the checkout and the installed-prefix stand-in.
    review_by : str
        Review date written into the record.

    Returns
    -------
    tuple
        The checkout, the prefix and the loaded record.
    """
    checkout, prefix = tmp_path / "checkout", tmp_path / "prefix"
    checkout.mkdir()
    shutil.copytree(ROOT / "requirements", checkout / "requirements")
    lock = checkout / LOCK
    lock.parent.mkdir(parents=True)
    lock.write_bytes((ROOT / LOCK).read_bytes())
    table = checkout / EXCEPTION_FILE
    table.parent.mkdir(parents=True)
    table.write_text(unreleased_record(review_by))
    for argv in (
        ["git", "init", "--initial-branch=main", str(checkout)],
        ["git", "add", "requirements", LOCK],
    ):
        subprocess.run(argv, cwd=checkout, capture_output=True, check=True, timeout=30)
    record = load_exceptions(table.read_bytes())[0]
    proof = prefix / record["proof_path"]
    proof.parent.mkdir(parents=True)
    proof.write_bytes(vendor_source(UNRELEASED_SOURCE, UNRELEASED_PROOF_SHA256))
    return checkout, prefix, record


@pytest.mark.parametrize(
    "change",
    ["none", "prerelease", "patch_release", "fixed_release", "bytes", "artifact", "fix", "unused"],
)
def test_unreleased_fix_holds_only_while_the_channel_has_nothing_newer(
    tmp_path: Path, change: str
) -> None:
    """Accept the hosted finding only while the exact release is installed and nothing newer exists.

    Parameters
    ----------
    tmp_path : Path
        Real independent Git checkout and an installed-prefix stand-in.
    change : str
        ``none`` and ``prerelease`` keep the exception valid, since a prerelease is not an
        installable fix. A newer patch or fixed release, altered installed bytes, another artifact,
        another fix boundary or a vanished finding each refuse it.
    """
    data = json.loads(UNRELEASED_RECORDED.read_text())
    match, artifact = data["match"], data["artifact"]
    assert artifact["metadata"]["sha256"] == UNRELEASED_ARTIFACT_SHA256
    checkout, prefix, record = unreleased_checkout(tmp_path, due_date())
    extra = {
        "prerelease": ("3.16.0rc1",),
        "patch_release": ("3.15.3",),
        "fixed_release": ("3.16.0",),
    }
    index = channel_index(*extra.get(change, ()))
    served: list[str] = []

    def read(url: str, limit: int) -> bytes:
        served.append(url)
        return index if url == CONDA_FORGE_PYTHON_INDEX else read_url(url, limit)

    actual = [copy.deepcopy(match)]
    inventories = {artifact["id"]: copy.deepcopy(artifact)}
    proof = prefix / record["proof_path"]
    if change == "bytes":
        proof.write_bytes(proof.read_bytes() + b"\n")
    elif change == "artifact":
        inventories[artifact["id"]]["metadata"]["sha256"] = "0" * 64
    elif change == "fix":
        actual[0]["vulnerability"]["fix"]["versions"] = ["3.15.3"]
    elif change == "unused":
        actual.clear()
    decision = conda_lock_decision(checkout, LOCK, actual, inventories, prefix, read)
    if change in {"none", "prerelease"}:
        assert served == [UNRELEASED_SOURCE, CONDA_FORGE_PYTHON_INDEX]
        assert decision["passed"] is True and decision["blocked"] == decision["unused"] == []
        assert decision["accepted"][0]["evidence"] == {
            "holds": True,
            "source": UNRELEASED_SOURCE,
            "proof_sha256": UNRELEASED_PROOF_SHA256,
            "channel": CONDA_FORGE_PYTHON_INDEX,
            "newer_releases": [],
        }
        return
    assert decision["passed"] is False and decision["accepted"] == []
    assert decision["blocked"] or decision["unused"]
    if change in {"patch_release", "fixed_release"}:
        assert decision["blocked"] == [match]


def test_oversized_channel_index_is_an_error(tmp_path: Path) -> None:
    """A channel document beyond the size limit stops the audit instead of passing it."""
    data = json.loads(UNRELEASED_RECORDED.read_text())
    checkout, prefix, _ = unreleased_checkout(tmp_path, due_date())

    def read(url: str, limit: int) -> bytes:
        return b"0" * limit if url == CONDA_FORGE_PYTHON_INDEX else read_url(url, limit)

    with pytest.raises(ValueError, match="size limit"):
        conda_lock_decision(
            checkout,
            LOCK,
            [data["match"]],
            {data["artifact"]["id"]: data["artifact"]},
            prefix,
            read,
        )


@pytest.mark.parametrize(
    ("index", "message"),
    [
        (b"[]", "no version list"),
        (b'{"versions": []}', "no version list"),
        (b'{"versions": [3.15]}', "not text"),
        (b'{"versions": ["3.15.2", "not a version"]}', "cannot be ordered"),
    ],
)
def test_unreadable_channel_index_is_refused(index: bytes, message: str) -> None:
    """Channel documents that cannot prove the absence of a newer release are refused.

    Parameters
    ----------
    index : bytes
        Malformed package document.
    message : str
        Expected refusal.
    """
    with pytest.raises(ValueError, match=message):
        newer_vendor_releases(index, "3.15.2")


def test_newer_releases_are_final_releases_in_order() -> None:
    """Only final releases newer than the installed version count, in ascending order."""
    index = channel_index("3.16.0", "3.15.3", "3.16.0rc1", "3.16.0.dev1", "3.14.9")
    assert newer_vendor_releases(index, "3.15.2") == ["3.15.3", "3.16.0"]
    assert newer_vendor_releases(channel_index(), "3.15.2") == []


def test_live_channel_index_is_readable() -> None:
    """The real conda-forge document parses and lists the installed release."""
    index = read_url(CONDA_FORGE_PYTHON_INDEX, 32 * 1024 * 1024 + 1)
    assert len(index) <= 32 * 1024 * 1024
    assert "3.15.2" in json.loads(index)["versions"]
    assert all(isinstance(version, str) for version in newer_vendor_releases(index, "3.15.2"))


@pytest.mark.parametrize(
    ("original", "replacement", "message"),
    [
        ('version = "3.15.2"', 'version = "3.16.0"', "already contains the vendor fix"),
        ('advisory = "CVE-2026-103995"', 'advisory = "CVE-2025-15367"', "vendor source"),
        ("/Lib/http/client.py", "/Lib/poplib.py", "vendor source"),
        ('reason = "vendor_fix_unreleased"', 'reason = "fixed_code_present"', "vendor source"),
    ],
)
def test_unreleased_fix_scope_cannot_be_broadened(
    original: str, replacement: str, message: str
) -> None:
    """The unreleased-fix reason admits only its advisory, its source and an unfixed version.

    Parameters
    ----------
    original : str
        Exact field of the committed record.
    replacement : str
        Candidate that removes a required boundary.
    message : str
        Expected refusal.
    """
    raw = unreleased_record(due_date())
    assert original in raw and len(load_exceptions(raw.encode())) == 1
    with pytest.raises(ValueError, match=message):
        load_exceptions(raw.replace(original, replacement).encode())


def test_unreleased_fix_review_cannot_exceed_two_weeks() -> None:
    """A review date beyond fourteen days is refused; fourteen days is accepted."""
    today = datetime.now(timezone.utc).date()
    assert load_exceptions(unreleased_record((today + timedelta(days=14)).isoformat()).encode())
    with pytest.raises(ValueError, match="exceeds two weeks"):
        load_exceptions(unreleased_record((today + timedelta(days=15)).isoformat()).encode())


def test_table_rewritten_during_verification_is_an_error(tmp_path: Path) -> None:
    """A table that changes while its proofs are read is refused rather than trusted."""
    data = json.loads(UNRELEASED_RECORDED.read_text())
    checkout, prefix, _ = unreleased_checkout(tmp_path, due_date())
    table = checkout / EXCEPTION_FILE

    def read(url: str, limit: int) -> bytes:
        table.write_text(table.read_text() + "\n")
        return channel_index() if url == CONDA_FORGE_PYTHON_INDEX else read_url(url, limit)

    with pytest.raises(ValueError, match="changed during its verification"):
        conda_lock_decision(
            checkout,
            LOCK,
            [data["match"]],
            {data["artifact"]["id"]: data["artifact"]},
            prefix,
            read,
        )
