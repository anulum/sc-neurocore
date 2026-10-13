# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Catalogue artifact recovery through the publication command

"""Recover missing views and refuse real filesystem publication failures."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests.test_record_readiness_receipt import (
    ARTIFACTS,
    RECEIPTS,
    VALIDATOR,
    assert_views_current,
    checkout_environment,
    recording_checkout as recording_checkout,
    run_publisher,
)


def test_refresh_recovers_a_missing_view_without_recording(recording_checkout: Path) -> None:
    """Recovery recreates a deleted derived view without another evidence run."""
    before = set((recording_checkout / RECEIPTS).glob("*.json"))
    missing = recording_checkout / ARTIFACTS[1]
    missing.unlink()
    result = run_publisher(recording_checkout, "refresh")
    assert result.returncode == 0, result.stdout + result.stderr
    assert missing.is_file()
    assert set((recording_checkout / RECEIPTS).glob("*.json")) == before
    assert_views_current(recording_checkout)


def test_refresh_public_api_keeps_the_same_catalogue_views(recording_checkout: Path) -> None:
    """Use the public Python entry point to refresh the same receipt-derived files."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'from tools.record_readiness_receipt import main; raise SystemExit(main(["refresh"]))',
        ],
        cwd=recording_checkout,
        env=checkout_environment(recording_checkout),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert_views_current(recording_checkout)


def test_refresh_refuses_a_directory_before_replacing_other_views(recording_checkout: Path) -> None:
    """An unusable destination leaves every other catalogue view untouched."""
    blocked = recording_checkout / ARTIFACTS[1]
    blocked.unlink()
    blocked.mkdir()
    profile = recording_checkout / ARTIFACTS[0]
    profile.write_bytes(profile.read_bytes() + b"\n")
    before = {
        path: (recording_checkout / path).read_bytes() for path in (ARTIFACTS[0], ARTIFACTS[2])
    }
    result = run_publisher(recording_checkout, "refresh")
    assert result.returncode == 2
    assert "not a regular file" in result.stderr
    assert {path: (recording_checkout / path).read_bytes() for path in before} == before


def test_staging_failure_does_not_replace_a_view(recording_checkout: Path) -> None:
    """A real missing output directory fails without replacing the Studio seal."""
    generated = recording_checkout / "docs/_generated"
    retained = generated.with_name("retained_generated")
    generated.rename(retained)
    generated.write_text("not a directory", encoding="utf-8")
    seal = recording_checkout / ARTIFACTS[2]
    before = seal.read_bytes()
    try:
        result = run_publisher(recording_checkout, "refresh")
        assert result.returncode == 2
        assert "could not refresh" in result.stderr
        assert seal.read_bytes() == before
    finally:
        generated.unlink()
        retained.rename(generated)


def test_existing_ownership_sidecar_is_preserved(recording_checkout: Path) -> None:
    """The public sidecar writer refuses an existing receipt's ownership file."""
    receipt = next((recording_checkout / RECEIPTS).glob("*.json.license"))
    before = receipt.read_bytes()
    original = receipt.with_name(receipt.name.removesuffix(".license"))
    script = (
        "from pathlib import Path; "
        "from tools.readiness_artifact_refresh import ReadinessArtifactError, write_receipt_license; "
        "import sys\n"
        "try:\n    write_receipt_license(Path(sys.argv[1]))\n"
        "except ReadinessArtifactError:\n    sys.exit(2)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(original)],
        cwd=recording_checkout,
        env=checkout_environment(recording_checkout),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert receipt.read_bytes() == before


def test_cosim_floor_refuses_a_changed_shared_compiler(recording_checkout: Path) -> None:
    """Withdraw the four verified profiles after a real change to their compiler subject."""
    source_root = Path(__file__).resolve().parents[1]
    source = source_root / "tests/test_readiness_evidence_ledger.py"
    target = recording_checkout / "tests/test_readiness_evidence_ledger.py"
    target.write_bytes(source.read_bytes())
    compiler = recording_checkout / "src/sc_neurocore/neurons/equation_safety.py"
    compiler.write_bytes(compiler.read_bytes() + b"\n")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "tests/test_readiness_evidence_ledger.py::test_recorded_cosim_profiles_keep_their_verified_labels",
        ],
        cwd=recording_checkout,
        env=checkout_environment(recording_checkout),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "co-simulation receipts lost their current source binding" in result.stdout
    for name in (
        "AdaptiveThresholdIFNeuron",
        "BrunelWangNeuron",
        "LapicqueNeuron",
        "ResonateAndFireNeuron",
    ):
        assert name in result.stdout


def test_record_failure_keeps_its_receipt_for_refresh_recovery(recording_checkout: Path) -> None:
    """Retain actual executed evidence when publication fails, then recover without a rerun."""
    blocked = recording_checkout / ARTIFACTS[1]
    original = blocked.read_bytes()
    blocked.unlink()
    blocked.mkdir()
    before = set((recording_checkout / RECEIPTS).glob("*.json"))
    result = run_publisher(
        recording_checkout,
        "record",
        "--model",
        "McCullochPittsNeuron",
        "--facet",
        "class_validated",
        "--",
        sys.executable,
        "-m",
        "pytest",
        VALIDATOR,
        "-q",
        "-p",
        "no:cacheprovider",
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "recorded receipts are retained" in result.stderr
    created = set((recording_checkout / RECEIPTS).glob("*.json")) - before
    assert len(created) == 1
    receipt = created.pop()
    retained_bytes = receipt.read_bytes()
    assert receipt.with_name(receipt.name + ".license").is_file()
    blocked.rmdir()
    blocked.write_bytes(original)
    recovered = run_publisher(recording_checkout, "refresh")
    assert recovered.returncode == 0, recovered.stdout + recovered.stderr
    assert set((recording_checkout / RECEIPTS).glob("*.json")) == before | {receipt}
    assert receipt.read_bytes() == retained_bytes
    assert_views_current(recording_checkout)
