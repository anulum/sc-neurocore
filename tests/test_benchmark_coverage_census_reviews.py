# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — benchmark census source-change review tests

"""A record whose bound sources changed only in documentation may carry a reviewed reason, and no more.

A review names a record, one of its bound sources, the digest of the source's current bytes and a reason. The census
accepts it only when the source still has those bytes and its code equals the code of the bytes the record measured,
which it finds in the repository's history. An accepted review makes the record ``reviewed``, not ``current``; a
review whose source's code changed is rejected and shown, and a review that applies to nothing is listed, so a
reason can neither hide a measurement that is owed nor linger after one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tools.benchmark_coverage_census import build_census, record_currency
from tools.source_change_reviews import REVIEWS_FILE, load_reviews

_SOURCE = "engine/src/kernel.rs"
_OLD = "/// Kernel.\npub fn step(v: i32) -> i32 {\n    v / 2 // half\n}\n"
_RECORD = "bench_adex.json"


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@localhost", *args],
        cwd=root,
        check=True,
        capture_output=True,
    )


def _repository(root: Path, *, commit: bool = True) -> Path:
    """Write a record that binds one Rust source, and commit both when asked; return the source.

    Parameters
    ----------
    root : Path
        The folder that stands for the repository root.
    commit : bool
        Whether to make it a Git repository with one commit.
    """
    source = root / _SOURCE
    source.parent.mkdir(parents=True)
    source.write_text(_OLD, encoding="utf-8")
    record = root / "benchmarks" / "results" / _RECORD
    record.parent.mkdir(parents=True)
    record.write_text(
        json.dumps({"model": "AdExNeuron", "source_hashes": {_SOURCE: _sha(_OLD.encode())}}),
        encoding="utf-8",
    )
    if commit:
        _git(root, "init", "-q")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", "record")
    return source


def _review(root: Path, source_bytes: bytes, *, record: str = _RECORD, path: str = _SOURCE) -> None:
    """Write a reviews file with one review of ``path`` in ``record`` at ``source_bytes``.

    Parameters
    ----------
    root : Path
        The repository root.
    source_bytes : bytes
        The source bytes the review was made on.
    record : str
        The reviewed record, below the results directory.
    path : str
        The reviewed source path.
    """
    reviews = root / REVIEWS_FILE
    reviews.parent.mkdir(parents=True, exist_ok=True)
    reviews.write_text(
        "schema_version = 1\n\n[[review]]\n"
        f'record = "{record}"\npath = "{path}"\nsource_sha256 = "{_sha(source_bytes)}"\n'
        'reason = "Only Rust comments changed; the code tokens equal the measured ones."\n',
        encoding="utf-8",
    )


def test_a_comment_change_without_a_review_leaves_the_record_outdated(tmp_path: Path) -> None:
    """Changed bytes are a change until someone reviews them."""
    source = _repository(tmp_path)
    source.write_text(_OLD.replace("// half", "// the half"), encoding="utf-8")
    assert record_currency(_RECORD, tmp_path) == {
        "name": _RECORD,
        "currency": "outdated",
        "changed": [_SOURCE],
    }


def test_a_reviewed_comment_change_makes_the_record_reviewed(tmp_path: Path) -> None:
    """The source's code equals the measured code and the review names its bytes, so the reason stands."""
    source = _repository(tmp_path)
    new = _OLD.replace("// half", "// the half").replace("/// Kernel.", "/// The kernel.")
    source.write_text(new, encoding="utf-8")
    _review(tmp_path, new.encode())
    assert record_currency(_RECORD, tmp_path) == {
        "name": _RECORD,
        "currency": "reviewed",
        "reviewed": [_SOURCE],
    }


def test_a_review_of_a_code_change_is_rejected_and_shown(tmp_path: Path) -> None:
    """A reason cannot stand in for a measurement once the code changed."""
    source = _repository(tmp_path)
    new = _OLD.replace("v / 2", "v / 3")
    source.write_text(new, encoding="utf-8")
    _review(tmp_path, new.encode())
    assert record_currency(_RECORD, tmp_path) == {
        "name": _RECORD,
        "currency": "outdated",
        "changed": [_SOURCE],
        "review_rejected": [_SOURCE],
    }


def test_a_review_of_other_bytes_is_rejected(tmp_path: Path) -> None:
    """A source changed again after its review is no longer the reviewed file."""
    source = _repository(tmp_path)
    reviewed = _OLD.replace("// half", "// the half")
    _review(tmp_path, reviewed.encode())
    source.write_text(reviewed.replace("// the half", "// a half"), encoding="utf-8")
    assert record_currency(_RECORD, tmp_path)["review_rejected"] == [_SOURCE]


def test_without_history_the_measured_bytes_cannot_be_found_and_the_review_is_rejected(
    tmp_path: Path,
) -> None:
    """Outside a repository the census cannot compare with the measured bytes, so it does not accept the reason."""
    source = _repository(tmp_path, commit=False)
    new = _OLD.replace("// half", "// the half")
    source.write_text(new, encoding="utf-8")
    _review(tmp_path, new.encode())
    assert record_currency(_RECORD, tmp_path)["currency"] == "outdated"


def test_a_review_that_applies_to_nothing_is_listed(tmp_path: Path) -> None:
    """A review of a record whose sources are current is listed, so it can be removed."""
    _repository(tmp_path)
    _review(tmp_path, _OLD.encode())
    census = build_census(tmp_path)
    assert census["unused_source_change_reviews"] == [f"{_RECORD}:{_SOURCE}"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("schema_version = 2\n", "schema_version"),
        ('schema_version = 1\n[[review]]\nrecord = "a.json"\n', "source_sha256"),
        (
            'schema_version = 1\n[[review]]\nrecord = "a.json"\npath = "p.rs"\nsource_sha256 = "x"\nreason = "r"\n',
            "source_sha256",
        ),
        (
            "schema_version = 1\n"
            + 2
            * f'[[review]]\nrecord = "a.json"\npath = "p.rs"\nsource_sha256 = "{64 * "a"}"\nreason = "r"\n',
            "twice",
        ),
        (
            f'schema_version = 1\n[[review]]\nrecord = "a.json"\npath = "/etc/passwd"\nsource_sha256 = "{64 * "a"}"\n'
            'reason = "r"\n',
            "path must be a relative path inside the repository",
        ),
        (
            f'schema_version = 1\n[[review]]\nrecord = "../a.json"\npath = "p.rs"\nsource_sha256 = "{64 * "a"}"\n'
            'reason = "r"\n',
            "record must be a relative path inside the repository",
        ),
    ],
    ids=[
        "a schema version",
        "a missing field",
        "a malformed digest",
        "a duplicate",
        "an absolute path",
        "a record outside the results",
    ],
)
def test_a_malformed_reviews_file_is_refused(tmp_path: Path, text: str, message: str) -> None:
    """The reviews file is read strictly, so a typo cannot silently drop or duplicate a review.

    Parameters
    ----------
    text : str
        The reviews file's text.
    message : str
        A word the refusal names.
    """
    reviews = tmp_path / REVIEWS_FILE
    reviews.parent.mkdir(parents=True)
    reviews.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_reviews(tmp_path)


def test_no_reviews_file_means_no_reviews(tmp_path: Path) -> None:
    """A tree without the file has no reviews, which is the state of every record before this."""
    assert load_reviews(tmp_path) == {}


def test_a_reviewed_source_beside_an_unreviewed_one_leaves_the_record_outdated(
    tmp_path: Path,
) -> None:
    """Each changed source needs its own accepted review; the reviewed one is listed apart."""
    source = _repository(tmp_path)
    other = tmp_path / "engine" / "src" / "other.rs"
    other.write_text("pub fn other() {}\n", encoding="utf-8")
    record = tmp_path / "benchmarks" / "results" / _RECORD
    payload = json.loads(record.read_text(encoding="utf-8"))
    payload["source_hashes"]["engine/src/other.rs"] = _sha(other.read_bytes())
    record.write_text(json.dumps(payload), encoding="utf-8")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "second source")
    new = _OLD.replace("// half", "// the half")
    source.write_text(new, encoding="utf-8")
    other.write_text("pub fn other() { () }\n", encoding="utf-8")
    _review(tmp_path, new.encode())
    assert record_currency(_RECORD, tmp_path) == {
        "name": _RECORD,
        "currency": "outdated",
        "changed": ["engine/src/other.rs"],
        "reviewed": [_SOURCE],
    }


def test_a_review_of_a_removed_source_is_rejected(tmp_path: Path) -> None:
    """A source that is gone has no bytes to review."""
    source = _repository(tmp_path)
    _review(tmp_path, _OLD.encode())
    source.unlink()
    assert record_currency(_RECORD, tmp_path)["review_rejected"] == [_SOURCE]


def test_a_review_is_rejected_when_no_commit_holds_the_measured_bytes(tmp_path: Path) -> None:
    """The record's digest must name bytes the history holds, or there is nothing to compare with."""
    source = _repository(tmp_path)
    record = tmp_path / "benchmarks" / "results" / _RECORD
    record.write_text(
        json.dumps({"model": "AdExNeuron", "source_hashes": {_SOURCE: 64 * "0"}}), encoding="utf-8"
    )
    _review(tmp_path, source.read_bytes())
    assert record_currency(_RECORD, tmp_path)["review_rejected"] == [_SOURCE]


def test_a_review_is_rejected_when_the_record_holds_no_digest_at_the_named_place(
    tmp_path: Path,
) -> None:
    """A manifest place the record lacks counts as a change, and it gives no measured bytes to compare with."""
    source = _repository(tmp_path)
    record = tmp_path / "benchmarks" / "results" / _RECORD
    record.write_text(json.dumps({"model": "AdExNeuron", "lanes": {}}), encoding="utf-8")
    _review(tmp_path, source.read_bytes())
    entry = record_currency(_RECORD, tmp_path, {_SOURCE: "lanes.kernel"})
    assert entry["currency"] == "outdated"
    assert entry["review_rejected"] == [_SOURCE]
