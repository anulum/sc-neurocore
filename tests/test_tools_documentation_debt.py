# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The cross-language documentation-debt measurement

"""A debt figure is only useful if it can be re-taken and is honest when it cannot.

The owner directive of 2026-09-06 asks for two things that pull against each
other: measure every language, and never let a heuristic become a number. The
resolution is that a language with no installed tool is recorded as *not
measured, with the reason* — and these cases hold that boundary, because the
tempting failure is to substitute a grep and report a plausible figure.

They also hold the shape of the artefact, since a record that cannot state the
tool, its version, the source it was taken against and the exact argv is an
anecdote rather than a measurement.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.documentation_debt import (
    DOCUMENTATION_DEBT_SCHEMA_VERSION,
    Measurement,
    build_report,
    main,
    measure_python,
    measure_rust,
    measure_typescript,
    read_eslint_report,
    read_ruff_concise,
    read_rustc_stderr,
    unmeasured,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestAnUnmeasuredLanguage:
    def test_it_carries_no_number_at_all(self) -> None:
        """A plausible figure from the wrong instrument reads as evidence."""
        entry = unmeasured("go", "revive", "no linter installed", ["a/b"])

        assert entry.undocumented is None
        assert entry.files is None
        assert entry.tool_version is None

    def test_it_states_the_reason_rather_than_leaving_it_blank(self) -> None:
        entry = unmeasured("julia", "Aqua.jl", "not in the shared environment", ["x"])

        assert entry.not_measured_reason == "not in the shared environment"
        assert entry.to_public_dict()["not_measured_reason"]

    def test_it_still_names_the_tool_that_would_measure_it(self) -> None:
        """Naming the tool is what makes the gap actionable rather than vague."""
        assert unmeasured("go", "revive", "reason", []).tool == "revive"


class TestTheArtefact:
    def test_it_records_what_the_figure_was_taken_against(self) -> None:
        report = build_report(
            [unmeasured("go", "revive", "no linter installed", ["a"])],
            source_sha="0" * 40,
        )

        assert report["schema_version"] == DOCUMENTATION_DEBT_SCHEMA_VERSION
        assert report["source_sha256"] == "0" * 40

    def test_it_counts_measured_and_unmeasured_languages_apart(self) -> None:
        """Summing an unmeasured language into a total would invent a figure."""
        report = build_report(
            [
                Measurement(
                    language="python",
                    tool="ruff",
                    tool_version="ruff 0.16.5",
                    argv=["ruff"],
                    undocumented=7,
                    files=3,
                    scopes=["src"],
                ),
                unmeasured("go", "revive", "no linter installed", ["a"]),
            ],
            source_sha="a" * 40,
        )

        assert report["summary"] == {
            "languages_measured": 1,
            "languages_not_measured": 1,
            "undocumented_total": 7,
        }

    def test_every_entry_carries_its_argv_so_it_can_be_re_taken(self) -> None:
        report = build_report(
            [
                Measurement(
                    language="python",
                    tool="ruff",
                    tool_version="ruff 0.16.5",
                    argv=["python", "-m", "ruff", "check"],
                    undocumented=1,
                    files=1,
                    scopes=["src"],
                )
            ],
            source_sha="b" * 40,
        )

        assert report["languages"][0]["argv"] == ["python", "-m", "ruff", "check"]


class TestPython:
    def test_it_counts_only_the_missing_documentation_rules(self, tmp_path: Path) -> None:
        """Style rules about an existing docstring are real, and are not this figure."""
        module = tmp_path / "sample.py"
        module.write_text(
            '"""Module docstring."""\n\n\ndef undocumented():\n    return 1\n',
            encoding="utf-8",
        )

        entry = measure_python(tmp_path, ["sample.py"])

        assert entry.tool_version is not None
        assert entry.undocumented == 1
        assert entry.files == 1
        assert "D103" in " ".join(entry.argv) or "D100" in " ".join(entry.argv)

    def test_a_documented_module_measures_zero(self, tmp_path: Path) -> None:
        module = tmp_path / "sample.py"
        module.write_text(
            '"""Module docstring."""\n\n\ndef documented() -> int:\n    """Return one."""\n    return 1\n',
            encoding="utf-8",
        )

        assert measure_python(tmp_path, ["sample.py"]).undocumented == 0


class TestToolsThatAreNotThere:
    def test_a_missing_eslint_config_is_reported_not_guessed(self, tmp_path: Path) -> None:
        (tmp_path / "studio" / "frontend").mkdir(parents=True)

        entry = measure_typescript(tmp_path, "absent.config.js")

        assert entry.undocumented is None
        assert "absent.config.js" in entry.not_measured_reason

    def test_a_missing_cargo_is_reported_not_guessed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tools.documentation_debt.shutil.which", lambda _name: None)

        entry = measure_rust(tmp_path, "engine/Cargo.toml")

        assert entry.undocumented is None
        assert entry.not_measured_reason == "cargo is not installed"


class TestTheCommandLine:
    def test_it_writes_an_artefact_naming_every_language(self, tmp_path: Path) -> None:
        target = tmp_path / "debt.json"

        assert (
            main(
                [
                    "--repo",
                    str(REPO_ROOT),
                    "--output",
                    str(target),
                    "--skip",
                    "python",
                    "--skip",
                    "typescript",
                    "--skip",
                    "rust",
                ]
            )
            == 0
        )

        report = json.loads(target.read_text(encoding="utf-8"))
        languages = {entry["language"] for entry in report["languages"]}
        assert languages == {"python", "typescript", "rust", "go", "julia", "mojo"}

    def test_a_skipped_language_is_not_silently_absent(self, tmp_path: Path) -> None:
        """A language missing from the report is indistinguishable from one with no debt."""
        target = tmp_path / "debt.json"
        main(
            [
                "--repo",
                str(REPO_ROOT),
                "--output",
                str(target),
                "--skip",
                "python",
                "--skip",
                "typescript",
                "--skip",
                "rust",
            ]
        )

        report = json.loads(target.read_text(encoding="utf-8"))
        skipped = [e for e in report["languages"] if e["language"] == "python"][0]
        assert skipped["undocumented"] is None
        assert skipped["not_measured_reason"] == "skipped in this run"

    def test_the_report_names_the_source_it_was_taken_against(self, tmp_path: Path) -> None:
        target = tmp_path / "debt.json"
        main(
            [
                "--repo",
                str(REPO_ROOT),
                "--output",
                str(target),
                "--skip",
                "python",
                "--skip",
                "typescript",
                "--skip",
                "rust",
            ]
        )

        report = json.loads(target.read_text(encoding="utf-8"))
        assert len(report["source_sha256"]) == 40


class TestReadingRuffOutput:
    def test_it_counts_one_line_per_missing_docstring(self) -> None:
        output = (
            "src/a.py:1:1: D100 Missing docstring in public module\n"
            "src/a.py:4:1: D103 Missing docstring in public function\n"
            "src/b.py:2:1: D101 Missing docstring in public class\n"
        )

        assert read_ruff_concise(output) == (3, 2)

    def test_a_style_rule_about_an_existing_docstring_is_not_counted(self) -> None:
        """D2xx and D4xx are real findings, and they are not this figure."""
        output = (
            "src/a.py:1:1: D100 Missing docstring in public module\n"
            "src/a.py:9:1: D205 1 blank line required between summary and description\n"
            "src/a.py:9:1: D401 First line should be in imperative mood\n"
        )

        assert read_ruff_concise(output) == (1, 1)

    def test_a_message_that_merely_mentions_a_code_is_not_counted(self) -> None:
        # The naive filter was `": D1" in line`, which this defeats.
        output = "src/a.py:3:1: E501 line too long (mentions D103 in a comment)\n"

        assert read_ruff_concise(output) == (0, 0)

    def test_no_findings_reads_as_zero_rather_than_failing(self) -> None:
        assert read_ruff_concise("") == (0, 0)


class TestReadingEslintOutput:
    def test_it_counts_only_the_missing_docblock_rule(self) -> None:
        report = [
            {
                "filePath": "/x/a.ts",
                "messages": [
                    {"ruleId": "jsdoc/require-jsdoc"},
                    {"ruleId": "jsdoc/require-returns"},
                ],
            },
            {"filePath": "/x/b.ts", "messages": [{"ruleId": "jsdoc/require-jsdoc"}]},
        ]

        assert read_eslint_report(report) == (2, 2)

    def test_a_clean_file_is_not_counted_as_a_file(self) -> None:
        report = [
            {"filePath": "/x/a.ts", "messages": []},
            {"filePath": "/x/b.ts", "messages": [{"ruleId": "jsdoc/require-jsdoc"}]},
        ]

        assert read_eslint_report(report) == (1, 1)

    def test_an_entry_without_messages_does_not_raise(self) -> None:
        assert read_eslint_report([{"filePath": "/x/a.ts"}]) == (0, 0)


class TestReadingRustcOutput:
    def test_it_counts_warnings_and_takes_files_from_the_location_lines(self) -> None:
        stderr = (
            "warning: missing documentation for a struct field\n"
            "   --> engine/src/wong_wang.rs:217:5\n"
            "warning: missing documentation for a struct field\n"
            "   --> engine/src/wong_wang.rs:218:5\n"
            "warning: missing documentation for a function\n"
            "   --> engine/src/lib.rs:10:1\n"
        )

        assert read_rustc_stderr(stderr) == (3, 2)

    def test_an_unrelated_warning_is_not_counted(self) -> None:
        stderr = (
            "warning: unused variable: `x`\n"
            "   --> engine/src/lib.rs:3:9\n"
            "warning: missing documentation for a function\n"
            "   --> engine/src/lib.rs:10:1\n"
        )

        assert read_rustc_stderr(stderr)[0] == 1

    def test_a_clean_build_reads_as_zero(self) -> None:
        assert read_rustc_stderr("    Finished `dev` profile\n") == (0, 0)


def _completed(stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    """Return a finished process carrying the output a case wants read."""
    return subprocess.CompletedProcess(args=["tool"], returncode=0, stdout=stdout, stderr=stderr)


class TestTheRunnersAssembleWhatTheReadersReturn:
    """The tools are exercised for real elsewhere; this is the wiring around them."""

    def test_typescript_reports_the_figure_eslint_gave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "studio" / "frontend").mkdir(parents=True)
        (tmp_path / "studio" / "frontend" / "eslint.measure.js").write_text("", encoding="utf-8")
        monkeypatch.setattr("tools.documentation_debt._version", lambda *_a, **_k: "v10.10.0")
        monkeypatch.setattr(
            "tools.documentation_debt._run",
            lambda *_a, **_k: _completed(
                stdout=json.dumps(
                    [{"filePath": "/x/a.ts", "messages": [{"ruleId": "jsdoc/require-jsdoc"}]}]
                )
            ),
        )

        entry = measure_typescript(tmp_path, "eslint.measure.js")

        assert entry.undocumented == 1
        assert entry.files == 1
        assert entry.tool_version == "v10.10.0"

    def test_typescript_says_so_when_eslint_returns_nothing_parseable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A crashed tool must not be read as a clean surface."""
        (tmp_path / "studio" / "frontend").mkdir(parents=True)
        (tmp_path / "studio" / "frontend" / "eslint.measure.js").write_text("", encoding="utf-8")
        monkeypatch.setattr("tools.documentation_debt._version", lambda *_a, **_k: "v10.10.0")
        monkeypatch.setattr(
            "tools.documentation_debt._run", lambda *_a, **_k: _completed(stdout="boom")
        )

        entry = measure_typescript(tmp_path, "eslint.measure.js")

        assert entry.undocumented is None
        assert entry.not_measured_reason == "eslint produced no JSON report"

    def test_rust_reports_the_figure_rustc_gave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tools.documentation_debt.shutil.which", lambda _n: "/usr/bin/cargo")
        monkeypatch.setattr("tools.documentation_debt._version", lambda *_a, **_k: "rustc 1.98.1")
        monkeypatch.setattr(
            "tools.documentation_debt._run",
            lambda *_a, **_k: _completed(
                stderr=(
                    "warning: missing documentation for a function\n   --> engine/src/lib.rs:10:1\n"
                )
            ),
        )

        entry = measure_rust(tmp_path, "engine/Cargo.toml")

        assert entry.undocumented == 1
        assert entry.files == 1
        assert entry.tool_version == "rustc 1.98.1"

    def test_python_says_so_when_ruff_cannot_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("tools.documentation_debt._version", lambda *_a, **_k: None)

        entry = measure_python(tmp_path, ["src"])

        assert entry.undocumented is None
        assert entry.not_measured_reason == "ruff is not importable in this interpreter"

    def test_a_tool_that_cannot_be_launched_has_no_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tools.documentation_debt import _version

        def raise_oserror(*_a: object, **_k: object) -> object:
            raise OSError("no such tool")

        monkeypatch.setattr("tools.documentation_debt._run", raise_oserror)

        assert _version(["absent"], cwd=tmp_path) is None

    def test_a_tool_that_prints_nothing_has_no_version(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        from tools.documentation_debt import _version

        monkeypatch.setattr("tools.documentation_debt._run", lambda *_a, **_k: _completed())

        assert _version(["quiet"], cwd=tmp_path) is None


class TestTheRunPrintsWhatItFound:
    def test_it_names_every_language_measured_and_not(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        main(
            [
                "--repo",
                str(REPO_ROOT),
                "--output",
                str(tmp_path / "d.json"),
                "--skip",
                "python",
                "--skip",
                "typescript",
                "--skip",
                "rust",
            ]
        )

        out = capsys.readouterr().out
        for language in ("python", "typescript", "rust", "go", "julia", "mojo"):
            assert language in out
        assert "not measured" in out

    def test_it_measures_each_language_when_nothing_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The wiring, not the tools: each measurement is reached and reported.

        The two toolchain-backed measurements are replaced here so the Python
        suite does not acquire a dependency on node and cargo — a suite that
        skips when a toolchain is absent reports green for a run that measured
        nothing, which is the failure this whole tool exists to prevent.
        """
        target = tmp_path / "debt.json"
        monkeypatch.setattr(
            "tools.documentation_debt.measure_python",
            lambda *_a, **_k: Measurement(
                language="python",
                tool="ruff",
                tool_version="ruff 0.16.5",
                argv=["ruff"],
                undocumented=2,
                files=1,
                scopes=["src"],
            ),
        )
        monkeypatch.setattr(
            "tools.documentation_debt.measure_typescript",
            lambda *_a, **_k: Measurement(
                language="typescript",
                tool="eslint",
                tool_version="v10",
                argv=["eslint"],
                undocumented=3,
                files=2,
                scopes=["studio/frontend"],
            ),
        )
        monkeypatch.setattr(
            "tools.documentation_debt.measure_rust",
            lambda *_a, **_k: Measurement(
                language="rust",
                tool="rustc",
                tool_version="rustc 1.98.1",
                argv=["cargo"],
                undocumented=5,
                files=4,
                scopes=["engine/Cargo.toml"],
            ),
        )

        assert main(["--repo", str(REPO_ROOT), "--output", str(target)]) == 0

        report = json.loads(target.read_text(encoding="utf-8"))
        assert report["summary"]["undocumented_total"] == 10
        assert report["summary"]["languages_measured"] == 3
        assert "undocumented in 1 files" in capsys.readouterr().out
