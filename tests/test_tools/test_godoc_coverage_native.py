# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — real Go parser protocol and declaration identity contracts

"""Exercise the compiled Go CLI and validate its actual individual findings."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from tests.test_tools_go_doc_ratchet import go_repository
from tools.go_doc_findings import measure_findings, parse_findings
from tools.go_doc_measurement import (
    GO_COVERAGE_TOOL,
    GoMeasurementError,
    go_scope,
    measure_go_coverage,
)

REPO = Path(__file__).resolve().parents[2]
_SOURCE = """// Package sample provides the declaration fixture.
package sample
// Left is a receiver.
type Left struct{}
// Right is a receiver.
type Right[T any] struct{}
func (Left) Record() {}
func (*Right[T]) Record() {}
func Fresh() {}
type Missing struct{}
const Limit = 1
var State = 2
"""


@pytest.fixture(scope="module")
def native_parser(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the actual Go CLI with native coverage instrumentation and no substitute."""
    binary = tmp_path_factory.mktemp("native-go-producer") / "godoc-coverage"
    subprocess.run(
        ["go", "build", "-cover", "-covermode=atomic", "-o", str(binary), GO_COVERAGE_TOOL],
        cwd=REPO,
        check=True,
        capture_output=True,
        timeout=120,
    )
    return binary


def _run(binary: Path, root: Path, paths: str, findings: Path) -> subprocess.CompletedProcess[str]:
    """Invoke the compiled native CLI with the literal source protocol input."""
    return subprocess.run(
        [str(binary), "-findings", str(findings)],
        cwd=root,
        input=paths,
        text=True,
        capture_output=True,
        check=False,
        timeout=15,
    )


def test_native_findings_distinguish_receivers_and_hash_parsed_bytes(
    tmp_path: Path, native_parser: Path
) -> None:
    """Actual parsed bytes and generic receiver identities accompany each declaration."""
    root = go_repository(tmp_path / "repo", {"sample/sample.go": _SOURCE})
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, "sample/sample.go", output)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    entries = json.loads(output.read_text())
    assert summary["schema_version"] == "sc-neurocore.go-doc-coverage.v3"
    assert summary["source_sha256"] == {
        "sample/sample.go": hashlib.sha256(_SOURCE.encode()).hexdigest()
    }
    assert {entry["receiver"] for entry in entries if entry["kind"] == "method"} == {
        "Left",
        "*Right[T]",
    }
    assert summary["by_kind"] == {"method": 2, "func": 1, "type": 1, "const": 1, "var": 1}
    cases = parse_findings(output.read_text(), ["sample/sample.go"], summary)
    assert len(cases) == 6
    declarations = parse_findings(
        json.dumps(summary["declarations"]), ["sample/sample.go"], summary, check_totals=False
    )
    assert cases <= declarations
    assert {case.name for case in declarations - cases} == {"Left", "Right", "sample"}
    measured = measure_findings(root, go_scope(root))
    assert len(measured.cases) == 6
    assert (
        measured.producer_sha256
        == hashlib.sha256((root / GO_COVERAGE_TOOL).read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    "paths",
    [
        "",
        "\n",
        " sample.go",
        "sample.go ",
        "sample.go\r\n",
        "sample.go\tsuffix.go",
        "../sample.go",
        "/sample.go",
        "./sample.go",
        "back\\slash.go",
        "sample.txt",
        "sample.go\nsample.go",
    ],
)
def test_native_scope_refuses_ambiguous_or_repeated_paths(
    tmp_path: Path, native_parser: Path, paths: str
) -> None:
    """The direct native entry point refuses trimming, traversal and repeated input."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, paths, output)
    assert result.returncode != 0 and not result.stdout
    assert not output.exists()


@pytest.mark.parametrize(
    "case", ["syntax", "missing", "directory", "symlink", "parent_symlink", "empty_receiver"]
)
def test_native_source_failures_never_emit_partial_figures(
    tmp_path: Path, native_parser: Path, case: str
) -> None:
    """Unreadable, aliased and malformed actual sources cannot yield a summary."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    path = "sample.go"
    if case == "syntax":
        (root / path).write_text("package !!!", encoding="utf-8")
    elif case == "empty_receiver":
        (root / path).write_text("package sample\nfunc () Record() {}\n", encoding="utf-8")
    elif case == "missing":
        path = "missing.go"
    elif case == "directory":
        path = "directory.go"
        (root / path).mkdir()
    elif case == "symlink":
        path = "alias.go"
        (root / path).symlink_to(root / "sample.go")
    else:
        (root / "alias").symlink_to(root, target_is_directory=True)
        path = "alias/sample.go"
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, path, output)
    assert result.returncode != 0 and not result.stdout, result.stdout + result.stderr
    assert not output.exists()
    assert "panic" not in result.stderr


def test_documented_native_sources_write_an_empty_array(
    tmp_path: Path, native_parser: Path
) -> None:
    """A measured zero carries an empty finding array and the source byte hash."""
    source = "// Package sample is documented.\npackage sample\n// Answer returns one.\nfunc Answer() int { return 1 }\n"
    root = go_repository(tmp_path / "repo", {"sample.go": source})
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, "\nsample.go\n\n", output)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text()) == []
    summary = json.loads(result.stdout)
    assert not parse_findings(output.read_text(), ["sample.go"], summary)


def test_native_findings_write_failure_refuses_the_summary(
    tmp_path: Path, native_parser: Path
) -> None:
    """A real output-directory error stops publication of the corresponding figure."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    result = _run(native_parser, root, "sample.go", tmp_path)
    assert result.returncode != 0 and not result.stdout


@pytest.mark.parametrize(
    "change",
    [
        "json",
        "array",
        "entry",
        "missing",
        "text",
        "line_bool",
        "line_zero",
        "path",
        "kind",
        "receiver",
        "package",
        "duplicate",
        "total",
        "files",
        "kinds",
    ],
)
def test_individual_protocol_rejects_corrupted_actual_records(
    tmp_path: Path, native_parser: Path, change: str
) -> None:
    """Native individual output cannot be replaced with an inconsistent receipt."""
    root = go_repository(tmp_path / "repo", {"sample.go": "package sample\nfunc Answer() {}\n"})
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, "sample.go", output)
    assert result.returncode == 0, result.stderr
    entries = json.loads(output.read_text())
    summary = json.loads(result.stdout)
    if change == "entry":
        entries[0] = None
    elif change == "missing":
        del entries[0]["name"]
    elif change == "text":
        entries[0]["name"] = ""
    elif change == "line_bool":
        entries[0]["line"] = True
    elif change == "line_zero":
        entries[0]["line"] = 0
    elif change == "path":
        entries[0]["file"] = "missing.go"
    elif change == "kind":
        entries[0]["kind"] = "unknown"
    elif change == "receiver":
        entries[0]["receiver"] = "Unexpected"
    elif change == "package":
        next(entry for entry in entries if entry["kind"] == "package")["name"] = "Other"
    elif change == "duplicate":
        entries.append(entries[0].copy())
    elif change == "total":
        summary["undocumented"] += 1
    elif change == "files":
        summary["files_with_findings"] += 1
    elif change == "kinds":
        summary["by_kind"] = {}
    raw = "{" if change == "json" else "{}" if change == "array" else json.dumps(entries)
    with pytest.raises(GoMeasurementError):
        parse_findings(raw, ["sample.go"], summary)


def test_individual_measurement_refuses_missing_native_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unavailable actual Go toolchain yields no individual measurement."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    paths = go_scope(root)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(GoMeasurementError, match="could not be read"):
        measure_findings(root, paths)


def test_individual_measurement_refuses_failed_native_toolchain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A real rejected Go configuration cannot produce a usable version receipt."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    monkeypatch.setenv("GOTOOLCHAIN", "unsupported")
    with pytest.raises(GoMeasurementError, match="version could not be qualified"):
        measure_findings(root, go_scope(root))


def test_individual_measurement_refuses_real_source_syntax_failure(tmp_path: Path) -> None:
    """Actual parser failure is propagated through the individual-measurement API."""
    root = go_repository(tmp_path / "repo", {"sample.go": "package !!!"})
    with pytest.raises(GoMeasurementError, match="command failed"):
        measure_findings(root, go_scope(root))


def test_native_package_and_group_comments_cover_their_actual_scope(
    tmp_path: Path, native_parser: Path
) -> None:
    """Shared package comments and declaration groups preserve native Go doc semantics."""
    first = """package sample
import "fmt"
func hidden() { fmt.Println("fixture") }
// Answer is documented.
func Answer() {}
type private int
type (
 // Public is documented at its specification.
 Public int
 lower int
)
// Bounds document every declaration in this group.
const (Low = 1; High = 2)
const (
 // Separate documents this constant specification.
 Separate = 3
 lowerConstant = 4
)
// Values documents this variable group.
var (First = 1; Second = 2)
var (
 // Third documents this variable specification.
 Third = 3
 lowerVariable = 4
 _ = 5
)
"""
    second = "// Package sample documents the shared package.\npackage sample\nfunc Debt() {}\n"
    external = "package sample_test\nfunc External() {}\n"
    root = go_repository(
        tmp_path / "repo",
        {"sample/a.go": first, "sample/b.go": second, "sample/external_test.go": external},
    )
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, "sample/a.go\nsample/b.go\nsample/external_test.go", output)
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["undocumented"] == 3 and summary["packages_total"] == 2
    assert summary["packages_undocumented"] == 1
    assert (
        len(
            parse_findings(
                output.read_text(),
                ["sample/a.go", "sample/b.go", "sample/external_test.go"],
                summary,
            )
        )
        == 3
    )


def test_native_scope_scanner_failure_never_measures_a_partial_list(
    tmp_path: Path, native_parser: Path
) -> None:
    """An oversized actual protocol token cannot truncate the submitted denominator."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    output = tmp_path / "findings.json"
    result = _run(native_parser, root, "sample.go\n" + "x" * (4 * 1024 * 1024 + 1), output)
    assert result.returncode != 0 and not result.stdout
    assert "reading the file list" in result.stderr and not output.exists()


def test_native_summary_output_failure_is_reported(tmp_path: Path, native_parser: Path) -> None:
    """An actual read-only stdout descriptor cannot be reported as successful output."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    output = tmp_path / "stdout.json"
    output.write_bytes(b"")
    with output.open("rb") as sink:
        result = subprocess.run(
            [str(native_parser)],
            cwd=root,
            input="sample.go",
            text=True,
            stdout=sink,
            stderr=subprocess.PIPE,
            check=False,
            timeout=15,
        )
    assert result.returncode != 0 and output.read_bytes() == b""


def test_actual_compiled_source_hash_fault_is_refused(tmp_path: Path) -> None:
    """A compiled mutation of the real parser cannot claim hashes of different bytes."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    producer = root / GO_COVERAGE_TOOL
    source = producer.read_text()
    original = "digest := sha256.Sum256(data)"
    assert source.count(original) == 1
    producer.write_text(
        source.replace(original, "digest := sha256.Sum256(append(data, '\\n'))"), encoding="utf-8"
    )
    paths = go_scope(root)
    with pytest.raises(GoMeasurementError, match="parsed bytes disagree"):
        measure_go_coverage(root, paths)
    with pytest.raises(GoMeasurementError, match="parsed different source bytes"):
        measure_findings(root, paths)


@pytest.mark.parametrize("replacement", ["nil", "[]finding{}"])
def test_actual_compiled_missing_declarations_are_refused(tmp_path: Path, replacement: str) -> None:
    """A real producer that loses retained declarations cannot qualify its debt count."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    producer = root / GO_COVERAGE_TOOL
    source = producer.read_text()
    original = "Declarations:         allDeclarations,"
    assert source.count(original) == 1
    producer.write_text(source.replace(original, "Declarations:         " + replacement + ","))
    paths = go_scope(root)
    with pytest.raises(GoMeasurementError, match="retain all exported"):
        measure_go_coverage(root, paths)
    with pytest.raises(GoMeasurementError, match="retain all exported"):
        measure_findings(root, paths)


def test_actual_compiled_findings_outside_declarations_are_refused(tmp_path: Path) -> None:
    """Real native output cannot replace a debt identity with another retained name."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    producer = root / GO_COVERAGE_TOOL
    source = producer.read_text()
    original = "byKind := map[string]int{}"
    assert source.count(original) == 1
    producer.write_text(
        source.replace(
            original,
            'for index := range allDeclarations { if allDeclarations[index].Kind == "func" { allDeclarations[index].Name += "Changed" } }\n\t'
            + original,
        )
    )
    with pytest.raises(GoMeasurementError, match="belong to the retained declaration cohort"):
        measure_findings(root, go_scope(root))


@pytest.mark.parametrize("fault", ["rewrite", "remove"])
def test_actual_external_producer_drift_is_refused(tmp_path: Path, fault: str) -> None:
    """Real compiled parser faults cannot change or remove the observed external producer."""
    root = go_repository(tmp_path / "repo", {"sample.go": _SOURCE})
    external = tmp_path / "external"
    producer = external / GO_COVERAGE_TOOL
    producer.parent.mkdir(parents=True)
    source = (root / GO_COVERAGE_TOOL).read_text()
    path = json.dumps(str(producer))
    mutation = (
        f"os.Remove({path})"
        if fault == "remove"
        else f'os.WriteFile({path}, []byte("changed producer bytes"), 0o644)'
    )
    assert source.count("func main() {") == 1
    producer.write_text(
        source.replace("func main() {", "func main() {\n\t" + mutation), encoding="utf-8"
    )
    expected = "no longer readable" if fault == "remove" else "producer changed"
    with pytest.raises(GoMeasurementError, match=expected):
        measure_findings(root, go_scope(root), producer_root=external)
    assert (
        not producer.exists()
        if fault == "remove"
        else producer.read_text() == "changed producer bytes"
    )
