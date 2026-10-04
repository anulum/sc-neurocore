# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — original native Go debt and source-cohort contracts

"""Exercise individual debt protection through real Git originals and native Go."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def local_git_fixture_context(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep private native Git fixtures separate from the hosting job's checkout identity.

    Individual CI-event tests supply their own explicit event context after
    this fixture. Real Git sources and native producer executions are retained.
    """
    for name in ("GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_EVENT_PATH", "GITHUB_EVENT_NAME"):
        monkeypatch.delenv(name, raising=False)


from tests.test_tools_go_doc_ratchet import go_repository
from tools.go_doc_findings import measure_findings
from tools.go_doc_history import LEGACY_CEILING_SCHEMA, protect_debt
from tools.go_doc_measurement import GoMeasurementError, go_scope

REPO = Path(__file__).resolve().parents[2]
_PACKAGE = "// Package sample supplies the test API.\npackage sample\n"
_OLD = _PACKAGE + "func Old() {}\n"


def _git(root: Path, *args: str) -> str:
    """Run actual fixture Git commands and require successful native completion."""
    return subprocess.check_output(["git", *args], cwd=root, text=True, timeout=10).strip()


def _baseline(root: Path, source: str = _OLD) -> tuple[Path, Path]:
    """Commit a real native-measured original source and its aggregate ceiling."""
    go_repository(root, {"sample/sample.go": source})
    original = measure_findings(root, go_scope(root))
    ceiling = root / "ceiling.json"
    ceiling.write_text(
        json.dumps(
            {
                "schema_version": LEGACY_CEILING_SCHEMA,
                "undocumented": original.measurement.undocumented,
                "files": original.measurement.files,
                "provenance": {"go": original.measurement.version},
            }
        ),
        encoding="utf-8",
    )
    _git(root, "add", "--", "ceiling.json")
    _git(root, "commit", "-qm", "original native ceiling")
    return root, ceiling


def _cli(root: Path, ceiling: Path, *, update: bool = False) -> subprocess.CompletedProcess[str]:
    """Check actual CLI and public API parity against independently committed originals."""
    argv = [
        sys.executable,
        "-B",
        "-m",
        "tools.go_doc_ratchet",
        "--repo",
        str(root),
        "--ceiling",
        str(ceiling),
    ]
    if update:
        argv.append("--update")
    result = subprocess.run(
        argv,
        cwd=REPO,
        env=os.environ.copy(),
        capture_output=True,
        text=True,
        check=False,
        timeout=90,
    )
    current = measure_findings(root, go_scope(root))
    if result.returncode == 2:
        with pytest.raises(GoMeasurementError):
            protect_debt(root, current, ceiling, allow_create=update)
    else:
        protect_debt(root, current, ceiling, allow_create=update)
    return result


def test_original_declarations_and_source_bytes_are_preserved(tmp_path: Path) -> None:
    """An unchanged declaration is allowed with reproducible original native evidence."""
    root, ceiling = _baseline(tmp_path / "repo")
    current = measure_findings(root, go_scope(root))
    protected = protect_debt(root, current, ceiling)
    assert protected.revision == _git(root, "rev-parse", "HEAD")
    assert protected.original.cases == current.cases
    assert protected.original_ceiling == 1
    record = protected.to_public_dict()
    assert record["original_ceiling_sha256"] is not None
    assert record["original"] == protected.original.to_public_dict()
    assert _cli(root, ceiling).returncode == 0


@pytest.mark.parametrize("case", ["function", "receiver", "package"])
def test_same_count_cannot_trade_original_debt(tmp_path: Path, case: str) -> None:
    """Equal aggregate counts cannot authorize a different function, receiver or package."""
    source = _OLD
    replacement = _PACKAGE + "// Old is documented.\nfunc Old() {}\nfunc New() {}\n"
    if case == "receiver":
        source = _PACKAGE + "// Old is a receiver.\ntype Old struct{}\nfunc (*Old) Record() {}\n"
        replacement = source.replace("Old", "New")
    elif case == "package":
        source = "package sample\n// Old is documented.\nfunc Old() {}\n"
        replacement = source.replace("package sample", "package replacement")
    root, ceiling = _baseline(tmp_path / "repo", source)
    (root / "sample/sample.go").write_text(replacement, encoding="utf-8")
    original_record = ceiling.read_bytes()
    actual = measure_findings(root, go_scope(root))
    assert actual.measurement.undocumented == 1
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "New undocumented Go declarations" in result.stdout
    assert ceiling.read_bytes() == original_record


def test_original_source_cannot_leave_the_git_discovered_cohort(tmp_path: Path) -> None:
    """Removing an original source cannot turn its disappeared debt into progress."""
    root, ceiling = _baseline(tmp_path / "repo")
    _git(root, "rm", "--", "sample/sample.go")
    result = _cli(root, ceiling)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "source cohort must not shrink" in result.stdout


@pytest.mark.parametrize("documented", [False, True])
def test_new_untracked_source_is_measured_immediately(tmp_path: Path, documented: bool) -> None:
    """A new file enters native measurement and requires documentation from creation."""
    root, ceiling = _baseline(tmp_path / "repo")
    source = _PACKAGE + ("// Fresh is documented.\n" if documented else "") + "func Fresh() {}\n"
    (root / "sample/fresh.go").write_text(source, encoding="utf-8")
    assert "sample/fresh.go" in go_scope(root)
    result = _cli(root, ceiling)
    assert result.returncode == (0 if documented else 2), result.stdout + result.stderr


def test_package_debt_identity_survives_first_file_selection(tmp_path: Path) -> None:
    """An earlier new file cannot rename a package-level original debt case."""
    root, ceiling = _baseline(
        tmp_path / "repo", "package sample\n// Old is documented.\nfunc Old() {}\n"
    )
    (root / "sample/aaa.go").write_text(
        "package sample\n// Fresh is documented.\nfunc Fresh() {}\n", encoding="utf-8"
    )
    result = _cli(root, ceiling)
    assert result.returncode == 0, result.stdout + result.stderr


def test_update_records_cases_and_forbids_their_reintroduction(tmp_path: Path) -> None:
    """A v2 lowered allowance refuses restored debt even before another commit."""
    root, ceiling = _baseline(tmp_path / "repo")
    (root / "sample/sample.go").write_text(
        _PACKAGE + "// Old is documented.\nfunc Old() {}\n", encoding="utf-8"
    )
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(ceiling.read_text())
    assert data["schema_version"] == "sc-neurocore.go-doc-ceiling.v2"
    assert data["provenance"]["debt_cases"] == []
    assert data["provenance"]["original_baseline"]["git_revision"] == _git(
        root, "rev-parse", "HEAD"
    )
    (root / "sample/sample.go").write_text(_OLD, encoding="utf-8")
    restored = _cli(root, ceiling)
    assert restored.returncode == 2 and "New undocumented" in restored.stdout


def test_original_ceiling_cannot_be_inflated(tmp_path: Path) -> None:
    """An edited candidate allowance cannot exceed the actual committed scalar."""
    root, ceiling = _baseline(tmp_path / "repo")
    data = json.loads(ceiling.read_text())
    data["undocumented"] = 2
    ceiling.write_text(json.dumps(data), encoding="utf-8")
    result = _cli(root, ceiling)
    assert result.returncode == 2 and "original Git ceiling" in result.stdout


@pytest.mark.parametrize("change", ["scalar", "version", "cases", "downgrade"])
def test_original_native_ceiling_must_reproduce(tmp_path: Path, change: str) -> None:
    """Original scalar, producer version and individual records require native proof."""
    root, ceiling = _baseline(tmp_path / "repo")
    if change in {"cases", "downgrade"}:
        assert _cli(root, ceiling, update=True).returncode == 0
    data = json.loads(ceiling.read_text())
    if change == "scalar":
        data["undocumented"] = 0
    elif change == "version":
        data["provenance"]["go"] = "go version incompatible"
    elif change == "cases":
        data["provenance"]["debt_cases"][0]["name"] = "Invented"
    ceiling.write_text(json.dumps(data), encoding="utf-8")
    _git(root, "add", "--", "ceiling.json")
    _git(root, "commit", "-qm", "original record under test")
    if change == "downgrade":
        data["schema_version"] = LEGACY_CEILING_SCHEMA
        ceiling.write_text(json.dumps(data), encoding="utf-8")
    result = _cli(root, ceiling)
    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize(
    "change",
    [
        "json",
        "object",
        "boolean",
        "negative",
        "schema",
        "case_array",
        "case_shape",
        "case_type",
        "duplicate",
        "count",
        "new_allowance",
    ],
)
def test_invalid_candidate_allowances_are_refused(tmp_path: Path, change: str) -> None:
    """Malformed or fabricated individual records never replace original debt proof."""
    root, ceiling = _baseline(tmp_path / "repo")
    assert _cli(root, ceiling, update=True).returncode == 0
    data = json.loads(ceiling.read_text())
    cases = data["provenance"]["debt_cases"]
    if change == "boolean":
        data["undocumented"] = True
    elif change == "negative":
        data["undocumented"] = -1
    elif change == "schema":
        data["schema_version"] = "unsupported"
    elif change == "case_array":
        data["provenance"]["debt_cases"] = None
    elif change == "case_shape":
        del cases[0]["receiver"]
    elif change == "case_type":
        cases[0]["name"] = 1
    elif change == "duplicate":
        cases.append(cases[0].copy())
    elif change == "count":
        cases.clear()
    elif change == "new_allowance":
        cases[0]["name"] = "Invented"
    raw = "{" if change == "json" else "[]" if change == "object" else json.dumps(data)
    ceiling.write_text(raw, encoding="utf-8")
    retained = ceiling.read_bytes()
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 2, result.stdout + result.stderr
    assert ceiling.read_bytes() == retained


def test_actual_ci_event_selects_the_original_source_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real push event protects pre-push debt instead of trusting its candidate HEAD."""
    root, ceiling = _baseline(tmp_path / "repo")
    before = _git(root, "rev-parse", "HEAD")
    (root / "sample/sample.go").write_text(_OLD.replace("Old", "New"), encoding="utf-8")
    _git(root, "add", "--", "sample/sample.go")
    _git(root, "commit", "-qm", "candidate debt trade")
    after = _git(root, "rev-parse", "HEAD")
    event = tmp_path / "event.json"
    event.write_text(json.dumps({"before": before, "after": after}), encoding="utf-8")
    for name, value in {
        "GITHUB_ACTIONS": "true",
        "GITHUB_SHA": after,
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_EVENT_PATH": str(event),
    }.items():
        monkeypatch.setenv(name, value)
    result = _cli(root, ceiling)
    assert result.returncode == 2 and "New undocumented" in result.stdout


def test_missing_original_git_revision_is_not_an_allowance(tmp_path: Path) -> None:
    """A real repository without a committed original cannot initialize a ceiling."""
    root, ceiling = _baseline(tmp_path / "repo")
    (root / ".git/refs/heads/master").unlink(missing_ok=True)
    (root / ".git/refs/heads/main").unlink(missing_ok=True)
    current = measure_findings(root, go_scope(root))
    with pytest.raises(GoMeasurementError, match="source revision"):
        protect_debt(root, current, ceiling)


def test_git_namespace_environment_cannot_substitute_source_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An actual foreign Git directory cannot replace the selected repository revision."""
    root, ceiling = _baseline(tmp_path / "repo")
    other, _ = _baseline(
        tmp_path / "foreign", _PACKAGE + "// Other is documented.\nfunc Other() {}\n"
    )
    own_revision = _git(root, "rev-parse", "HEAD")
    foreign_revision = _git(other, "rev-parse", "HEAD")
    assert own_revision != foreign_revision
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(other))
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 0, result.stdout + result.stderr
    record = json.loads(ceiling.read_text())
    assert record["provenance"]["git_revision"] == own_revision
    assert record["provenance"]["original_baseline"]["git_revision"] == own_revision


def test_missing_candidate_can_only_initialize_from_original_debt(tmp_path: Path) -> None:
    """Explicit initialization measures original cases and preserves their scalar ceiling."""
    root, ceiling = _baseline(tmp_path / "repo")
    ceiling.unlink()
    current = measure_findings(root, go_scope(root))
    with pytest.raises(GoMeasurementError, match="no ceiling record"):
        protect_debt(root, current, ceiling)
    protected = protect_debt(root, current, ceiling, allow_create=True)
    assert protected.original_ceiling == 1 and protected.original.cases == current.cases


def test_original_symlink_source_is_not_qualified_as_a_regular_file(tmp_path: Path) -> None:
    """A current replacement cannot conceal an original Git symlink source artifact."""
    root, ceiling = _baseline(tmp_path / "repo")
    alias = root / "sample/alias.go"
    alias.symlink_to("sample.go")
    _git(root, "add", "--", "sample/alias.go")
    _git(root, "commit", "-qm", "original symlink artifact")
    alias.unlink()
    alias.write_text(_PACKAGE + "// Fresh is documented.\nfunc Fresh() {}\n", encoding="utf-8")
    current = measure_findings(root, go_scope(root))
    with pytest.raises(GoMeasurementError, match="regular repository files"):
        protect_debt(root, current, ceiling)


def test_removing_an_original_function_does_not_repay_documentation(tmp_path: Path) -> None:
    """Retain original ABI debt declarations while allowing actual documentation repayment."""
    root, ceiling = _baseline(tmp_path / "repo")
    original = ceiling.read_bytes()
    (root / "sample/sample.go").write_text(_PACKAGE, encoding="utf-8")
    current = measure_findings(root, go_scope(root))
    assert current.measurement.undocumented == 0
    with pytest.raises(GoMeasurementError, match="declarations must remain"):
        protect_debt(root, current, ceiling)
    result = _cli(root, ceiling, update=True)
    assert result.returncode == 2 and "declarations must remain" in result.stdout
    assert ceiling.read_bytes() == original
