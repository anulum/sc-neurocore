# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark variant source digest tests

"""A record's per-variant source digests are checked at each variant's own revision.

A record that compares a parent revision with a candidate hashes each variant's sources inside
``variants.<i>.source_hashes``. A parent's digests describe the parent revision, so they cannot be checked at the
record's commit; a clean variant is checked at its ``git_head``, a parent at its ``source_ref``, and a variant measured
on a dirty tree at the record's commit, the only commit that can hold the bytes it hashed.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from tools.benchmark_evidence_gate import (
    _git_bytes,
    _safe_repo_path,
    evaluate_benchmark_evidence_gate,
)
from tools.benchmark_variant_source_hashes import (
    VARIANT_REVISION_MODE,
    variant_revision,
    variant_source_hash_failures,
)

_OLD, _NEW, _UNCOMMITTED = b"value = 1\n", b"value = 2\n", b"value = 3\n"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return done.stdout.strip()


def _history(repo: Path) -> tuple[str, str, str]:
    """Commit ``src/a.py`` twice and then a record; return the two source commits and the record's.

    Parameters
    ----------
    repo : Path
        An empty folder for the repository.
    """
    _git(repo, "init", "-q")
    source = repo / "src" / "a.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(_OLD)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "old")
    old = _git(repo, "rev-parse", "HEAD")
    source.write_bytes(_NEW)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "new")
    new = _git(repo, "rev-parse", "HEAD")
    (repo / "record.json").write_text("{}\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "record")
    return old, new, _git(repo, "rev-parse", "HEAD")


def _record(old: str, new: str, *, parent: bytes = _OLD, dirty: bytes = _NEW) -> dict[str, Any]:
    """Return a record with a parent, a clean candidate and a dirty candidate, each hashing ``src/a.py``.

    Parameters
    ----------
    old, new : str
        The first and second source commits.
    parent : bytes
        The bytes the parent's digest is taken of.
    dirty : bytes
        The bytes the dirty candidate's digest is taken of.
    """
    return {
        "variants": [
            {
                "git_head": None,
                "source_ref": old[:9],
                "surface_dirty": None,
                "source_hashes": {"src/a.py": _sha(parent)},
            },
            {
                "git_head": new,
                "source_ref": "candidate",
                "surface_dirty": False,
                "source_hashes": {"src/a.py": _sha(_NEW)},
            },
            {
                "git_head": old,
                "source_ref": "working-tree",
                "surface_dirty": True,
                "source_hashes": {"src/a.py": _sha(dirty)},
            },
        ]
    }


def _failures(repo: Path, record: Any, commit: str | None) -> list[tuple[str, str]]:
    return variant_source_hash_failures(
        record,
        repo_root=repo,
        record_commit=commit,
        git_bytes=_git_bytes,
        safe_path=_safe_repo_path,
    )


def test_every_variant_is_checked_at_its_own_revision(tmp_path: Path) -> None:
    """The parent at its source_ref, the clean candidate at its git_head, the dirty one at the record's commit."""
    old, new, record_commit = _history(tmp_path)
    assert _failures(tmp_path, _record(old, new), record_commit) == []


def test_a_parent_digest_of_other_bytes_is_stale(tmp_path: Path) -> None:
    """A parent digest that is not of the parent revision's bytes is refused at its place."""
    old, new, record_commit = _history(tmp_path)
    assert _failures(tmp_path, _record(old, new, parent=_NEW), record_commit) == [
        ("stale_variant_source_hash", "variants.0.source_hashes.src/a.py")
    ]


def test_a_revision_that_does_not_resolve_is_refused(tmp_path: Path) -> None:
    """A variant whose revision the history does not hold cannot be checked, so it fails."""
    old, new, record_commit = _history(tmp_path)
    record = _record(old, new)
    record["variants"][0]["source_ref"] = "no-such-revision"
    assert _failures(tmp_path, record, record_commit) == [
        ("unresolvable_variant_revision", "variants.0")
    ]


def test_a_dirty_variant_whose_bytes_the_record_commit_did_not_take_is_refused(
    tmp_path: Path,
) -> None:
    """The dirty tree's bytes must be the record's commit's, or the record cannot be verified."""
    old, new, record_commit = _history(tmp_path)
    assert _failures(tmp_path, _record(old, new, dirty=_UNCOMMITTED), record_commit) == [
        ("dirty_variant_not_committed", "variants.2.source_hashes.src/a.py")
    ]


def test_a_dirty_variant_is_not_checked_at_its_own_git_head(tmp_path: Path) -> None:
    """The dirty candidate's git_head holds the old bytes; only the record's commit holds what it hashed."""
    old, new, record_commit = _history(tmp_path)
    assert _git(tmp_path, "show", f"{old}:src/a.py") + "\n" == _OLD.decode()
    assert _failures(tmp_path, _record(old, new), record_commit) == []


def test_a_path_the_revision_lacks_or_an_unsafe_path_is_refused(tmp_path: Path) -> None:
    """A source missing at the variant's revision, and a path outside the repository, are both failures."""
    old, new, record_commit = _history(tmp_path)
    record = _record(old, new)
    record["variants"][1]["source_hashes"] = {"src/gone.py": _sha(_NEW), "../escape.py": _sha(_NEW)}
    assert _failures(tmp_path, record, record_commit) == [
        ("invalid_source_path", "variants.1.source_hashes.../escape.py"),
        ("missing_historical_source", "variants.1.source_hashes.src/gone.py"),
    ]


def test_a_variant_that_names_no_revision_is_refused(tmp_path: Path) -> None:
    """A clean variant without a git_head or a source_ref names nothing to check against."""
    old, new, record_commit = _history(tmp_path)
    record = _record(old, new)
    record["variants"][0]["source_ref"] = None
    assert _failures(tmp_path, record, record_commit) == [
        ("unresolvable_variant_revision", "variants.0")
    ]


@pytest.mark.parametrize(
    ("record", "expected"),
    [
        ({}, [("missing_variants", "variants")]),
        ({"variants": []}, [("missing_variants", "variants")]),
        (
            {"variants": [{"git_head": "x"}, "label"]},
            [
                ("variant_without_source_hashes", "variants.0"),
                ("variant_without_source_hashes", "variants.1"),
            ],
        ),
    ],
    ids=["no variants", "an empty list", "variants without digests"],
)
def test_a_record_without_variant_digests_is_refused(
    tmp_path: Path, record: dict[str, Any], expected: list[tuple[str, str]]
) -> None:
    """A gate that asks for variant checks needs variants that hold digests.

    Parameters
    ----------
    record : dict
        The record.
    expected : list
        The failures.
    """
    assert _failures(tmp_path, record, None) == expected


def test_without_a_record_commit_a_dirty_variant_is_checked_against_the_files(
    tmp_path: Path,
) -> None:
    """Under the worktree policy a dirty variant is compared with the source as it is on disk."""
    old, new, _record_commit = _history(tmp_path)
    assert _failures(tmp_path, _record(old, new), None) == []
    (tmp_path / "src" / "a.py").unlink()
    assert _failures(tmp_path, _record(old, new), None) == [
        ("missing_historical_source", "variants.2.source_hashes.src/a.py")
    ]


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ({"git_head": "abc", "source_ref": "def"}, "abc"),
        ({"git_head": None, "source_ref": "def"}, "def"),
        ({"git_head": "", "source_ref": "def"}, "def"),
        ({"git_head": None, "source_ref": None}, None),
        ({}, None),
    ],
    ids=["git_head", "a parent", "an empty git_head", "neither", "no fields"],
)
def test_a_clean_variant_is_read_at_its_git_head_and_a_parent_at_its_source_ref(
    variant: dict[str, Any], expected: str | None
) -> None:
    """``git_head`` names the revision a clean variant ran; a parent names its revision in ``source_ref``.

    Parameters
    ----------
    variant : dict
        The variant.
    expected : str or None
        The revision it names.
    """
    assert variant_revision(variant) == expected


def _gate_repo(
    repo: Path,
    *,
    mode: str = VARIANT_REVISION_MODE,
    parent: bytes = _OLD,
    policy: str = "artefact_commit",
) -> Path:
    """Commit a record with variants and a manifest whose gate asks for variant checks; return the manifest.

    Parameters
    ----------
    repo : Path
        An empty folder for the repository.
    mode : str
        The gate's ``variant_source_hashes`` value.
    parent : bytes
        The bytes the parent's digest is taken of.
    policy : str
        The manifest's source digest policy.
    """
    repo.mkdir(parents=True, exist_ok=True)
    old, new, _commit = _history(repo)
    record = {"latency_ns": 1.0, **_record(old, new, parent=parent)}
    (repo / "benchmarks" / "results").mkdir(parents=True)
    (repo / "benchmarks" / "results" / "demo.json").write_text(json.dumps(record), encoding="utf-8")
    manifest = repo / "benchmarks" / "benchmark_regression_gates.json"
    manifest.write_text(
        json.dumps(
            {
                "SPDX-License-Identifier": "AGPL-3.0-or-later",
                "schema_version": "sc-neurocore.benchmark-regression-gates.v1",
                "source_hash_policy": policy,
                "gates": [
                    {
                        "id": "demo",
                        "artefact": "benchmarks/results/demo.json",
                        "required_numbers": ["latency_ns"],
                        "variant_source_hashes": mode,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "measure")
    return manifest


def test_the_gate_checks_the_variants_of_a_gate_that_asks(tmp_path: Path) -> None:
    """Through the gate, a correct record passes and a wrong parent digest is refused under the gate's id."""
    manifest = _gate_repo(tmp_path / "good")
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "good.json", repo_root=tmp_path / "good"
    )
    assert report["passed"], report["failures"]
    manifest = _gate_repo(tmp_path / "bad", parent=_NEW)
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "bad.json", repo_root=tmp_path / "bad"
    )
    assert report["failures"] == [
        {
            "gate_id": "demo",
            "reason": "stale_variant_source_hash",
            "path": "variants.0.source_hashes.src/a.py",
        }
    ]


def test_the_gate_refuses_an_unknown_variant_mode(tmp_path: Path) -> None:
    """Only ``variant_revision`` is a mode, so a typo cannot switch the check off unnoticed."""
    manifest = _gate_repo(tmp_path, mode="variant_revisions")
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "report.json", repo_root=tmp_path
    )
    assert report["failures"] == [
        {
            "gate_id": "demo",
            "reason": "variant_source_hashes_has_unknown_mode",
            "path": "variant_source_hashes",
        }
    ]


def test_a_record_changed_after_its_commit_is_reported_once_and_its_variants_are_not_checked(
    tmp_path: Path,
) -> None:
    """Without a source map the variant check finds the record's commit itself, and stops when it cannot."""
    manifest = _gate_repo(tmp_path)
    record = tmp_path / "benchmarks" / "results" / "demo.json"
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["variants"][0]["source_hashes"]["src/a.py"] = _sha(_NEW)
    record.write_text(json.dumps(payload), encoding="utf-8")
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "report.json", repo_root=tmp_path
    )
    assert report["failures"] == [
        {
            "gate_id": "demo",
            "reason": "artefact_differs_from_commit",
            "path": "benchmarks/results/demo.json",
        }
    ]


def test_under_the_worktree_policy_a_dirty_variant_is_compared_with_the_files(
    tmp_path: Path,
) -> None:
    """A local run without the commit policy still checks every variant, the dirty one against the source on disk."""
    manifest = _gate_repo(tmp_path, policy="worktree")
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "good.json", repo_root=tmp_path
    )
    assert report["passed"], report["failures"]
    (tmp_path / "src" / "a.py").write_bytes(_UNCOMMITTED)
    report = evaluate_benchmark_evidence_gate(
        manifest_path=manifest, output_path=tmp_path / "bad.json", repo_root=tmp_path
    )
    assert report["failures"] == [
        {
            "gate_id": "demo",
            "reason": "dirty_variant_not_committed",
            "path": "variants.2.source_hashes.src/a.py",
        }
    ]
