# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Executed receipt publication and catalogue consistency

"""Publish real reference-validator results in disposable production checkouts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sc_neurocore.neurons.facet_receipts import credit_problems, load_receipt

REPO_ROOT = Path(__file__).resolve().parents[1]
RECEIPTS = Path("src/sc_neurocore/neurons/facet_receipts")
ARTIFACTS = (
    Path("docs/_generated/model_profile_ledger.json"),
    Path("docs/_generated/readiness_evidence_ledger.json"),
    Path("src/sc_neurocore/studio/verified_readiness.json"),
)
VALIDATOR = (
    "tests/test_reference_mcculloch_pitts.py::"
    "test_truth_rows_match_an_independent_absolute_veto_rule"
)


@pytest.fixture
def recording_checkout(tmp_path: Path) -> Iterator[Path]:
    """Provide a real detached checkout containing the candidate publication tools.

    Parameters
    ----------
    tmp_path : Path
        Pytest's owned temporary allocation.

    Yields
    ------
    Path
        Production checkout, removed after its bounded child commands finish.
    """
    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(checkout), "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        timeout=30,
    )
    try:
        for name in ("record_readiness_receipt.py", "readiness_artifact_refresh.py"):
            shutil.copyfile(REPO_ROOT / "tools" / name, checkout / "tools" / name)
        yield checkout
    finally:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(checkout)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            timeout=30,
        )


def checkout_environment(checkout: Path) -> dict[str, str]:
    """Bind child imports and temporary files to the disposable production tree.

    Parameters
    ----------
    checkout : Path
        Detached source checkout.

    Returns
    -------
    dict of str to str
        Child environment without bytecode or pytest cache writes.
    """
    environment = os.environ.copy()
    environment.update(
        PYTHONPATH=os.pathsep.join((str(checkout / "src"), str(checkout))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_ADDOPTS="",
        TMPDIR=str(checkout.parent),
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
    )
    return environment


def run_publisher(checkout: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    """Run the real publication command in its own process.

    Parameters
    ----------
    checkout : Path
        Detached source checkout.
    *arguments : str
        Literal publication command arguments.

    Returns
    -------
    subprocess.CompletedProcess[str]
        Actual command result, including diagnostics and exit status.
    """
    return subprocess.run(
        [sys.executable, "-u", str(checkout / "tools/record_readiness_receipt.py"), *arguments],
        cwd=checkout,
        env=checkout_environment(checkout),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def assert_views_current(checkout: Path) -> None:
    """Check all three views through their existing generator commands.

    Parameters
    ----------
    checkout : Path
        Detached source checkout after publication or recovery.
    """
    for tool in ("model_profile_ledger", "readiness_evidence_ledger", "studio_readiness_seal"):
        result = subprocess.run(
            [sys.executable, str(checkout / "tools" / f"{tool}.py"), "--check"],
            cwd=checkout,
            env=checkout_environment(checkout),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("collect_only", [False, True], ids=["passed", "not-executed"])
def test_record_updates_every_view_from_the_actual_validator(
    recording_checkout: Path, collect_only: bool
) -> None:
    """Passed and unexecuted validators both replace the catalogue's receipt references."""
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
        *(["--collect-only"] if collect_only else []),
    )
    assert result.returncode == (1 if collect_only else 0), result.stdout + result.stderr
    created = set((recording_checkout / RECEIPTS).glob("*.json")) - before
    assert len(created) == 1
    path = created.pop()
    receipt = load_receipt(path)
    assert bool(credit_problems(receipt)) == collect_only
    assert receipt.counts["passed"] == (0 if collect_only else 1)
    assert receipt.runtime["git_head"]
    assert all(old.is_file() for old in before)
    ownership = path.with_name(path.name + ".license").read_text(encoding="utf-8")
    # REUSE-IgnoreStart
    assert ownership.startswith("SPDX-License-Identifier: AGPL-3.0-or-later\n")
    # REUSE-IgnoreEnd
    assert "(concepts)" in ownership and "(code)" in ownership
    for artifact in ARTIFACTS:
        assert path.name in (recording_checkout / artifact).read_text(encoding="utf-8")
    assert_views_current(recording_checkout)


@pytest.mark.parametrize("arguments", [("--receipt-dir", "archive"), ("--receipt-dir=archive",)])
def test_publication_refuses_an_archive_store(
    recording_checkout: Path, arguments: tuple[str, ...]
) -> None:
    """A receipt written outside the catalogue cannot be presented as a published view."""
    before = {path: (recording_checkout / path).read_bytes() for path in ARTIFACTS}
    result = run_publisher(recording_checkout, "record", *arguments)
    assert result.returncode == 2
    assert "canonical receipt store" in result.stderr
    assert {path: (recording_checkout / path).read_bytes() for path in ARTIFACTS} == before


def test_unknown_model_keeps_existing_views_and_receipts(recording_checkout: Path) -> None:
    """Recorder refusal never starts a catalogue update or removes prior evidence."""
    before = {path: (recording_checkout / path).read_bytes() for path in ARTIFACTS}
    receipts = set((recording_checkout / RECEIPTS).glob("*.json"))
    result = run_publisher(
        recording_checkout,
        "record",
        "--model",
        "UnknownNeuron",
        "--facet",
        "class_validated",
        "--",
        sys.executable,
        "-m",
        "pytest",
        VALIDATOR,
    )
    assert result.returncode == 2
    assert "not a registered class" in result.stderr
    assert set((recording_checkout / RECEIPTS).glob("*.json")) == receipts
    assert {path: (recording_checkout / path).read_bytes() for path in ARTIFACTS} == before


@pytest.mark.parametrize("arguments", [("--help",), ()], ids=["help", "missing-action"])
def test_publication_help_names_recording_and_recovery(
    recording_checkout: Path, arguments: tuple[str, ...]
) -> None:
    """Describe both supported actions and refuse an invocation with no action."""
    result = run_publisher(recording_checkout, *arguments)
    assert result.returncode == (0 if arguments else 2)
    output = result.stdout + result.stderr
    assert "record" in output and "refresh" in output
