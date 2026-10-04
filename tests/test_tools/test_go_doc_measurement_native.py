# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go documentation cohort and source byte contracts

"""Exercise real Git, Go, public measurement APIs and private ceiling updates."""

from __future__ import annotations

import hashlib
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tools.documentation_debt import measure_go
from tools.go_doc_measurement import (
    GO_COVERAGE_TOOL,
    GoMeasurementError,
    capture_source,
    go_scope,
    measure_go_coverage,
    qualify_repository,
    validate_summary,
)

REPO = Path(__file__).resolve().parents[2]
_DOCUMENTED = "// Package sample supplies a documented API.\npackage sample\n// Answer returns one.\nfunc Answer() int { return 1 }\n"


@pytest.fixture(autouse=True)
def local_git_fixture_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep real private Git sources independent of the hosting job's checkout identity."""
    for name in ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_EVENT_PATH", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(name, raising=False)


def _project(root: Path) -> Path:
    """Create a native Git fixture holding the unchanged real parser and Go source."""
    assert shutil.which("go") is not None, "Native Go is required for these contracts"
    root.mkdir()
    (root / "sample.go").write_text(_DOCUMENTED, encoding="utf-8")
    shutil.copytree(REPO / "tools/godoc_coverage", root / "tools/godoc_coverage")
    for argv in (
        ["git", "init", "-q"],
        ["git", "add", "--", "sample.go", GO_COVERAGE_TOOL],
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(argv, cwd=root, check=True, capture_output=True, timeout=10)
    return root


def _cli(root: Path, ceiling: Path) -> subprocess.CompletedProcess[str]:
    """Run the production ratchet CLI without simulated producers or canonical writes."""
    return subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.go_doc_ratchet",
            "--repo",
            str(root),
            "--ceiling",
            str(ceiling),
            "--update",
        ],
        cwd=REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=45,
    )


def test_actual_parser_receipt_binds_current_working_bytes(tmp_path: Path) -> None:
    """Valid native figures bind byte changes even when the Git revision stays fixed."""
    root = _project(tmp_path / "repo")
    original = qualify_repository(root)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True)
    (root / "sample.go").write_text(
        "package sample\nfunc Answer() int { return 1 }\n", encoding="utf-8"
    )
    current = qualify_repository(root)
    assert original.undocumented == 0 and current.undocumented == 2
    assert current.files == 1
    assert current.source.scope_sha256 == original.source.scope_sha256
    assert current.source.source_sha256 != original.source.source_sha256
    assert subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True) == revision
    report = measure_go(root).to_public_dict()
    assert report["provenance"] == current.to_public_dict()
    assert measure_go_coverage(root, go_scope(root)) == current.summary


def test_actual_cli_update_records_native_input_manifest(tmp_path: Path) -> None:
    """A private update retains every measured path and the actual parser hash."""
    root = _project(tmp_path / "repo")
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text('{"undocumented":2}', encoding="utf-8")
    result = _cli(root, ceiling)
    assert result.returncode == 0, result.stdout + result.stderr
    provenance = json.loads(ceiling.read_text())["provenance"]
    capsule = provenance["measurement"]
    assert capsule["paths"] == go_scope(root)
    assert capsule["go"].startswith("go version go")
    assert len(provenance["git_revision"]) == 40
    assert len(capsule["source_sha256"]) == 64
    for path, digest in capsule["source_pins"].items():
        assert digest == hashlib.sha256((root / path).read_bytes()).hexdigest()


def test_retained_measurement_detects_actual_tracked_cohort_growth(tmp_path: Path) -> None:
    """A new native Git index entry invalidates the original tracked-source receipt."""
    root = _project(tmp_path / "repo")
    recorded = qualify_repository(root)
    recorded.verify(root, tracked=True)
    (root / "fresh.go").write_text(_DOCUMENTED, encoding="utf-8")
    subprocess.run(["git", "add", "--", "fresh.go"], cwd=root, check=True, timeout=10)
    with pytest.raises(GoMeasurementError, match="cohort changed"):
        recorded.verify(root, tracked=True)


def test_retained_record_version_is_checked_against_native_go(tmp_path: Path) -> None:
    """A controlled stale record is refused against the actual installed Go version."""
    root = _project(tmp_path / "repo")
    recorded = qualify_repository(root)
    stale = replace(recorded, version=recorded.version + " stale")
    with pytest.raises(GoMeasurementError, match="version changed"):
        stale.verify(root)


@pytest.mark.parametrize("files", [0, -1, True])
def test_summary_scope_count_requires_a_positive_integer(tmp_path: Path, files: int) -> None:
    """Native output cannot qualify an empty, negative or boolean scope count."""
    root = _project(tmp_path / "repo")
    recorded = qualify_repository(root)
    with pytest.raises(GoMeasurementError, match="positive integer"):
        validate_summary(json.dumps(recorded.summary), files=files)


@pytest.mark.parametrize("name", [" sample.go", "\tsample.go", "line\nbreak.go"])
def test_actual_tracked_ambiguous_path_refuses_update(tmp_path: Path, name: str) -> None:
    """Actual Git paths cannot be trimmed or split into another parser input."""
    root = _project(tmp_path / "repo")
    (root / name).write_text("package hidden\nfunc Hidden() int { return 2 }\n", encoding="utf-8")
    subprocess.run(["git", "add", "--", name], cwd=root, check=True, timeout=10)
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text('{"undocumented":10,"retain":"original"}', encoding="utf-8")
    retained = ceiling.read_bytes()
    result = _cli(root, ceiling)
    assert result.returncode == 2, result.stdout + result.stderr
    assert ceiling.read_bytes() == retained
    assert measure_go(root).undocumented is None


@pytest.mark.parametrize(
    "paths",
    [
        [],
        ["sample.go", "sample.go"],
        ["../sample.go"],
        ["/sample.go"],
        ["./sample.go"],
        ["sample.txt"],
        ["back\\slash.go"],
        [""],
    ],
)
def test_public_measurement_rejects_invalid_cohorts(tmp_path: Path, paths: list[str]) -> None:
    """Empty, repeated, escaping and protocol-incompatible sources yield no number."""
    root = _project(tmp_path / "repo")
    with pytest.raises(GoMeasurementError):
        measure_go_coverage(root, paths)


@pytest.mark.parametrize("change", ["source", "parser", "new_config", "removed_source"])
def test_snapshot_refuses_observed_native_input_changes(tmp_path: Path, change: str) -> None:
    """The production verifier detects actual byte, producer and module input drift."""
    root = _project(tmp_path / "repo")
    snapshot = capture_source(root, go_scope(root))
    snapshot.verify(root)
    if change == "new_config":
        (root / "go.mod").write_text("module changed\n", encoding="utf-8")
    elif change == "removed_source":
        (root / "sample.go").unlink()
    else:
        path = root / ("sample.go" if change == "source" else GO_COVERAGE_TOOL)
        path.write_text(path.read_text() + "\n// Actual input change.\n", encoding="utf-8")
    with pytest.raises(GoMeasurementError):
        snapshot.verify(root)


def test_symbolic_source_cannot_escape_input_custody(tmp_path: Path) -> None:
    """An actual symlink into a different source is refused before native measurement."""
    root = _project(tmp_path / "repo")
    (root / "alias.go").symlink_to(root / "sample.go")
    with pytest.raises(GoMeasurementError, match="symbolic"):
        measure_go_coverage(root, ["alias.go"])


@pytest.mark.parametrize("case", ["failed_git", "failed_go", "syntax", "undecodable"])
def test_actual_native_failures_preserve_private_ceiling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    """Broken Git, invalid native Go configuration and real syntax errors fail closed."""
    root = _project(tmp_path / "repo")
    if case == "failed_git":
        (root / ".git/config").write_text("[broken\n", encoding="utf-8")
    elif case == "failed_go":
        monkeypatch.setenv("GOTOOLCHAIN", "unsupported")
    elif case == "undecodable":
        name = os.fsdecode(b"invalid\xff.go")
        (root / name).write_text(_DOCUMENTED, encoding="utf-8")
        subprocess.run(["git", "add", "--", name], cwd=root, check=True, timeout=10)
    else:
        (root / "sample.go").write_text("package !!!\n", encoding="utf-8")
    ceiling = tmp_path / "ceiling.json"
    ceiling.write_text('{"undocumented":10,"retain":"original"}', encoding="utf-8")
    retained = ceiling.read_bytes()
    result = _cli(root, ceiling)
    assert result.returncode == 2, result.stdout + result.stderr
    assert ceiling.read_bytes() == retained
    assert measure_go(root).undocumented is None


def test_missing_native_command_cannot_be_a_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An actual empty PATH refuses native Git and Go without substituting executables."""
    root = _project(tmp_path / "repo")
    paths = go_scope(root)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GoMeasurementError):
        go_scope(root)
    with pytest.raises(GoMeasurementError):
        measure_go_coverage(root, paths)


def test_native_empty_git_cohort_stays_unmeasured(tmp_path: Path) -> None:
    """A repository without any tracked or new Go source yields no Go figure."""
    root = tmp_path / "empty"
    root.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=root, check=True, timeout=10)
    assert go_scope(root) == []
    assert measure_go(root).undocumented is None


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "boolean",
        "negative",
        "cohort",
        "kinds",
        "bad_kind",
        "kind_boolean",
        "kind_negative",
        "total",
        "packages",
        "package_debt",
        "affected",
        "zero_affected",
        "hashes_missing",
        "hash_count",
        "hash_type",
        "hash_value",
    ],
)
def test_native_summary_contract_rejects_inconsistent_records(tmp_path: Path, change: str) -> None:
    """Validate real parser output against corrupted protocol fields without fake tools."""
    root = _project(tmp_path / "repo")
    (root / "sample.go").write_text(
        "package sample\nfunc Answer() int { return 1 }\n", encoding="utf-8"
    )
    actual = qualify_repository(root)
    summary = actual.summary
    replacements: dict[str, tuple[str, object]] = {
        "schema": ("schema_version", "unsupported"),
        "boolean": ("undocumented", True),
        "negative": ("files_scanned", -1),
        "cohort": ("files_scanned", 1),
        "kinds": ("by_kind", []),
        "bad_kind": ("by_kind", {"unknown": 2}),
        "kind_boolean": ("by_kind", {"func": True}),
        "kind_negative": ("by_kind", {"func": -1}),
        "total": ("by_kind", {"func": 9, "package": 1}),
        "packages": ("packages_total", 0),
        "package_debt": ("packages_undocumented", 3),
        "affected": ("files_with_findings", 3),
        "zero_affected": ("files_with_findings", 0),
        "hashes_missing": ("source_sha256", None),
        "hash_count": ("source_sha256", {}),
        "hash_type": ("source_sha256", {"sample.go": True, GO_COVERAGE_TOOL: "0" * 64}),
        "hash_value": ("source_sha256", {"sample.go": "invalid", GO_COVERAGE_TOOL: "0" * 64}),
    }
    key, value = replacements[change]
    summary[key] = value
    with pytest.raises(GoMeasurementError):
        validate_summary(json.dumps(summary), files=len(actual.source.paths))


@pytest.mark.parametrize("raw", ["{", "[]"])
def test_summary_requires_an_actual_json_object(raw: str) -> None:
    """Malformed or nonobject records cannot serve as native parser summaries."""
    with pytest.raises(GoMeasurementError):
        validate_summary(raw, files=1)
