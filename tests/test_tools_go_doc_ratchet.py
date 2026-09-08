# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The Go documentation ratchet, exercised against real Go

"""What the Go documentation ratchet must do, stated as behaviour.

Every case that needs a figure gets it from the real tool over real Go source
written into a temporary tree: the point of this lane is that the measurement
comes from a parser rather than from a pattern, and a test that fed the ratchet
a number would not have checked the half that was wrong before.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from tools.doc_debt_ceiling import RatchetError
from tools.documentation_debt import go_scope, measure_go, measure_go_coverage
from tools.go_doc_ratchet import (
    GO_DOC_CEILING_SCHEMA_VERSION,
    compare,
    main,
    measure,
    write_ceiling,
)

pytestmark = pytest.mark.skipif(
    shutil.which("go") is None, reason="the Go toolchain is not installed"
)

PROVENANCE = {
    "argv": "git ls-files -- '*.go' | go run tools/godoc_coverage/main.go",
    "go": "go version go1.24.0 linux/amd64",
    "source_sha256": "0" * 40,
}

REPO_ROOT = Path(__file__).resolve().parents[1]

DOCUMENTED = """\
// Package sample is documented.
package sample

// Answer returns the answer.
func Answer() int { return 42 }
"""

UNDOCUMENTED_FUNC = """\
// Package sample is documented.
package sample

func Answer() int { return 42 }
"""


def go_repository(root: Path, files: dict[str, str]) -> Path:
    """Write a git repository holding the given Go sources and return its path.

    The scope the tool measures comes from ``git ls-files``, so the fixture has
    to be a real repository with the files really tracked; writing them to a
    directory would produce an empty scope and a test that proves nothing.
    """
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    shutil.copytree(REPO_ROOT / "tools" / "godoc_coverage", root / "tools" / "godoc_coverage")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
    return root


class TestTheMeasurementComesFromTheParser:
    def test_a_documented_package_and_function_leave_no_debt(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "clean", {"sample/sample.go": DOCUMENTED})

        undocumented, files, version = measure(root)

        assert undocumented == 0
        assert files == 0
        assert version.startswith("go version")

    def test_an_undocumented_exported_function_is_found(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "one", {"sample/sample.go": UNDOCUMENTED_FUNC})

        undocumented, files, _ = measure(root)

        assert undocumented == 1
        assert files == 1

    def test_an_unexported_function_is_not_debt(self, tmp_path: Path) -> None:
        source = (
            "// Package sample is documented.\npackage sample\n\nfunc answer() int { return 42 }\n"
        )
        root = go_repository(tmp_path / "unexported", {"sample/sample.go": source})

        assert measure(root)[0] == 0

    def test_a_package_comment_is_counted_once_for_the_package(self, tmp_path: Path) -> None:
        """Go puts one package comment on one file; per-file counting inflates it."""
        first = "package sample\n\n// A returns one.\nfunc A() int { return 1 }\n"
        second = "package sample\n\n// B returns two.\nfunc B() int { return 2 }\n"
        root = go_repository(tmp_path / "pkgdoc", {"sample/a.go": first, "sample/b.go": second})

        assert measure(root)[0] == 1

    def test_a_documented_group_documents_the_names_inside_it(self, tmp_path: Path) -> None:
        source = (
            "// Package sample is documented.\npackage sample\n\n"
            "// Limits are the bounds.\nconst (\n\tLow = 1\n\tHigh = 2\n)\n"
        )
        root = go_repository(tmp_path / "group", {"sample/sample.go": source})

        assert measure(root)[0] == 0

    def test_a_file_that_does_not_parse_stops_the_run(self, tmp_path: Path) -> None:
        """A skipped file shrinks the denominator, and that reads as progress."""
        root = go_repository(tmp_path / "broken", {"sample/sample.go": "package !!!"})

        with pytest.raises(RatchetError, match="did not produce a figure"):
            measure(root)


class TestTheScopeIsEveryTrackedFile:
    def test_a_new_file_nobody_registered_is_measured(self, tmp_path: Path) -> None:
        """The hole a list-shaped scope leaves: enforcement is the default here."""
        root = go_repository(tmp_path / "grow", {"sample/sample.go": DOCUMENTED})
        (root / "fresh").mkdir()
        (root / "fresh" / "fresh.go").write_text(
            "package fresh\n\nfunc New() int { return 0 }\n", encoding="utf-8"
        )
        subprocess.run(["git", "add", "fresh/fresh.go"], cwd=root, check=True)

        assert measure(root)[0] == 2

    def test_an_untracked_file_is_outside_the_scope(self, tmp_path: Path) -> None:
        """Stated because it bounds the claim: git's index is what is measured."""
        root = go_repository(tmp_path / "untracked", {"sample/sample.go": DOCUMENTED})
        (root / "loose.go").write_text(
            "package loose\n\nfunc X() int { return 0 }\n", encoding="utf-8"
        )

        assert measure(root)[0] == 0

    def test_the_scope_is_taken_from_git_not_from_a_walk(self, tmp_path: Path) -> None:
        """A walk of this tree reaches a vendored toolchain; git's index does not."""
        root = go_repository(tmp_path / "walk", {"sample/sample.go": DOCUMENTED})
        vendored = root / ".venv" / "lib" / "go"
        vendored.mkdir(parents=True)
        (vendored / "vendored.go").write_text(
            "package vendored\n\nfunc Y() int { return 0 }\n", encoding="utf-8"
        )

        # The fixture tracks the coverage tool too, because `go run` needs it there;
        # the vendored file is the one that must be absent.
        assert go_scope(root) == ["sample/sample.go", "tools/godoc_coverage/main.go"]


class TestTheRatchetOnlyTurnsOneWay:
    def test_a_rise_fails_and_says_by_how_much(self) -> None:
        verdict = compare(1533, 1532)

        assert verdict.ok is False
        assert "Go documentation debt rose" in verdict.summary()
        assert "+1" in verdict.summary()

    def test_a_fall_passes_and_asks_for_an_update(self) -> None:
        verdict = compare(1500, 1532)

        assert verdict.ok is True
        assert "--update" in verdict.summary()

    def test_it_refuses_to_raise_a_recorded_ceiling(self, tmp_path: Path) -> None:
        path = tmp_path / "ceiling.json"
        write_ceiling(path, undocumented=10, files=2, provenance=PROVENANCE)

        with pytest.raises(RatchetError, match="refusing to raise"):
            write_ceiling(path, undocumented=11, files=2, provenance=PROVENANCE)

    def test_the_record_carries_its_contract_and_what_produced_it(self, tmp_path: Path) -> None:
        path = tmp_path / "ceiling.json"
        write_ceiling(path, undocumented=10, files=2, provenance=PROVENANCE)

        document = json.loads(path.read_text(encoding="utf-8"))
        assert document["schema_version"] == GO_DOC_CEILING_SCHEMA_VERSION
        assert document["provenance"]["go"].startswith("go version")
        assert "may not rise" in document["note"]


class TestTheCommandLine:
    def test_it_fails_when_debt_rose_above_the_ceiling(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = go_repository(tmp_path / "cli-rise", {"sample/sample.go": UNDOCUMENTED_FUNC})
        ceiling = tmp_path / "ceiling.json"
        write_ceiling(ceiling, undocumented=0, files=0, provenance=PROVENANCE)

        assert main(["--repo", str(root), "--ceiling", str(ceiling)]) == 1
        assert "rose" in capsys.readouterr().out

    def test_it_passes_when_debt_is_at_the_ceiling(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "cli-level", {"sample/sample.go": UNDOCUMENTED_FUNC})
        ceiling = tmp_path / "ceiling.json"
        write_ceiling(ceiling, undocumented=1, files=1, provenance=PROVENANCE)

        assert main(["--repo", str(root), "--ceiling", str(ceiling)]) == 0

    def test_a_missing_ceiling_stops_the_check_rather_than_passing_it(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        root = go_repository(tmp_path / "cli-absent", {"sample/sample.go": DOCUMENTED})

        assert main(["--repo", str(root), "--ceiling", str(tmp_path / "absent.json")]) == 2
        assert "no ceiling record" in capsys.readouterr().out

    def test_update_lowers_the_ceiling_and_records_the_source(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "cli-update", {"sample/sample.go": DOCUMENTED})
        ceiling = tmp_path / "ceiling.json"
        write_ceiling(ceiling, undocumented=5, files=1, provenance=PROVENANCE)

        assert main(["--repo", str(root), "--ceiling", str(ceiling), "--update"]) == 0
        document = json.loads(ceiling.read_text(encoding="utf-8"))
        assert document["undocumented"] == 0
        assert len(document["provenance"]["source_sha256"]) == 40


class TestTheDebtReportNoLongerCallsGoUnmeasurable:
    def test_go_is_measured_with_a_figure_and_a_named_tool(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "report", {"sample/sample.go": UNDOCUMENTED_FUNC})

        measurement = measure_go(root)

        assert measurement.undocumented == 1
        assert measurement.not_measured_reason == ""
        assert "go/parser" in measurement.tool
        assert measurement.tool_version is not None

    def test_the_summary_carries_the_contract_it_claims(self, tmp_path: Path) -> None:
        root = go_repository(tmp_path / "schema", {"sample/sample.go": DOCUMENTED})

        summary = measure_go_coverage(root, go_scope(root))

        assert summary["schema_version"] == "sc-neurocore.go-doc-coverage.v1"
        # The fixture package and the coverage tool's own package: it measures itself.
        assert summary["packages_total"] == 2
