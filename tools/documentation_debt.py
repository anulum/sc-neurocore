#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-language documentation debt, measured reproducibly

"""How much of each language's surface carries no documentation.

The owner directive of 2026-09-06 requires documentation on the public surface
of **every** language a project ships, caught by a gate rather than noticed
later, and it is explicit about how the figure behind such a gate is produced:

* measure per language with a tool that understands that language;
* record the measurement reproducibly — tool version, language version, source
  digest, exact scopes and argv — because a figure that cannot be re-taken is
  an anecdote;
* set any ratchet at the measured ceiling, never at a copied or estimated one.

That last rule is why this exists. A regex over source lines is a heuristic: it
may size a problem, it may not become a ceiling. Measured on this repository
the difference was not small — a heuristic count of the TypeScript surface
reported 719 undocumented declarations where ESLint, the tool that actually
enforces the rule, reported 1297.

A language whose measurement tool is not installed is recorded as
``not_measured`` **with the reason**. It is never given a number from a
substitute method: a plausible figure from the wrong instrument is worse than
an honest gap, because it looks like evidence.

That reason has to keep being re-read, though, because it can name the wrong
thing. Go stood here as not measured for want of a Go documentation *linter* —
true, and beside the point: ``go/parser`` is what such linters are built on, it
ships with the toolchain, and the toolchain was installed the whole time. The
gap was in the reading, not in the language.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tools.go_doc_measurement import (
    GO_COVERAGE_SCHEMA_VERSION as GO_COVERAGE_SCHEMA_VERSION,
    GO_COVERAGE_TOOL as GO_COVERAGE_TOOL,
    GoMeasurementError,
    go_scope as go_scope,
    measure_go_coverage as measure_go_coverage,
    qualify_repository,
)
from tools.rust_doc_measurement import RustMeasurementError, measure_rust_findings


#: Contract version of the measurement artefact.
DOCUMENTATION_DEBT_SCHEMA_VERSION = "sc-neurocore.documentation-debt.v1"

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Measurement:
    """One language's documentation debt, or the reason it is not known.

    Attributes
    ----------
    language : str
        The language measured.
    tool : str
        The tool that produced the figure, or the tool that would.
    tool_version : str or None
        Exactly what ran; ``None`` when nothing did.
    argv : list of str
        The command, so the figure can be re-taken verbatim.
    undocumented : int or None
        Declarations carrying no documentation, or ``None`` when not measured.
    files : int or None
        Files carrying at least one of them, or ``None`` when not measured.
    scopes : list of str
        The paths the figure covers.
    not_measured_reason : str
        Empty when measured; otherwise why no figure exists.
    provenance : dict of str to object
        Native source evidence when the language producer qualifies its inputs.
    """

    language: str
    tool: str
    tool_version: str | None
    argv: list[str]
    undocumented: int | None
    files: int | None
    scopes: list[str] = field(default_factory=list)
    not_measured_reason: str = ""
    provenance: dict[str, object] = field(default_factory=dict)

    def to_public_dict(self) -> dict[str, Any]:
        """Return the JSON form recorded in the artefact."""
        result: dict[str, object] = {
            "argv": list(self.argv),
            "files": self.files,
            "language": self.language,
            "not_measured_reason": self.not_measured_reason,
            "scopes": list(self.scopes),
            "tool": self.tool,
            "tool_version": self.tool_version,
            "undocumented": self.undocumented,
        }
        if self.provenance:
            result["provenance"] = dict(self.provenance)
        return result


def _run(
    argv: Sequence[str], *, cwd: Path, timeout: float = 3600.0
) -> subprocess.CompletedProcess[str]:
    """Run a measurement command, returning its completed process."""
    return subprocess.run(  # noqa: S603 - fixed argv built here, no shell
        list(argv), capture_output=True, text=True, cwd=cwd, timeout=timeout, check=False
    )


def _version(argv: Sequence[str], *, cwd: Path) -> str | None:
    """Return a tool's own version string, or ``None`` when it cannot run."""
    try:
        completed = _run(argv, cwd=cwd, timeout=120.0)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    text = (completed.stdout or completed.stderr).strip().splitlines()
    return text[0] if text else None


def read_ruff_concise(output: str) -> tuple[int, int]:
    """Return the count and file count of ruff's missing-documentation lines.

    Ruff's concise format is ``path:line:col: CODE message``. Only the ``D1xx``
    codes say a declaration carries no docstring, and a line that merely
    mentions one in its message must not be counted.

    Parameters
    ----------
    output : str
        Ruff's concise stdout.

    Returns
    -------
    tuple of (int, int)
        How many declarations carry no docstring, and in how many files.
    """
    lines = [line for line in output.splitlines() if _is_missing_doc_line(line)]
    return len(lines), len({line.split(":", 1)[0] for line in lines})


def _is_missing_doc_line(line: str) -> bool:
    """Return whether one concise ruff line reports a missing docstring."""
    parts = line.split(": ", 1)
    if len(parts) != 2:
        return False
    code = parts[1].split(" ", 1)[0]
    return len(code) == 4 and code.startswith("D1") and code[2:].isdigit()


def read_eslint_report(report: Sequence[dict[str, Any]]) -> tuple[int, int]:
    """Return the count and file count of ESLint's missing-docblock findings.

    Only ``jsdoc/require-jsdoc`` is counted: it is the rule that says a
    declaration carries no docblock at all. Rules about an existing docblock's
    shape are real, and they are not this figure.

    Parameters
    ----------
    report : sequence of dict
        ESLint's JSON report, already parsed.

    Returns
    -------
    tuple of (int, int)
        How many declarations carry no docblock, and in how many files.
    """
    undocumented = 0
    files: set[str] = set()
    for entry in report:
        hits = [
            message
            for message in entry.get("messages", [])
            if message.get("ruleId") == "jsdoc/require-jsdoc"
        ]
        undocumented += len(hits)
        if hits:
            files.add(str(entry.get("filePath", "")))
    return undocumented, len(files)


def read_rustc_stderr(stderr: str) -> tuple[int, int]:
    """Return the count and file count of rustc's ``missing_docs`` warnings.

    Each warning is followed by a ``-->`` location line, so the files are taken
    from those rather than from the warning text, which names no path.

    Parameters
    ----------
    stderr : str
        Rustc's stderr, as cargo passes it through.

    Returns
    -------
    tuple of (int, int)
        How many items carry no documentation, and in how many files.
    """
    undocumented = 0
    files: set[str] = set()
    awaiting_location = False
    for line in stderr.splitlines():
        if line.startswith("warning:"):
            awaiting_location = line.startswith("warning: missing documentation")
            if awaiting_location:
                undocumented += 1
        elif awaiting_location and line.strip().startswith("--> "):
            location = line.strip()[4:]
            files.add(location.rsplit(":", 2)[0])
            awaiting_location = False
    return undocumented, len(files)


@dataclass(frozen=True, slots=True)
class _Attempt:
    """An attempted native command and the provenance retained if it cannot measure.

    Attributes
    ----------
    language : str
        Language whose declared source cohort was submitted.
    tool : str
        Native measurement tool required for that cohort.
    argv : list of str
        Command arguments for the attempted or refused invocation.
    scopes : list of str
        Declared source roots or manifest paths.
    tool_version : str or None
        Successful version-probe output, or None when it was unavailable.
    """

    language: str
    tool: str
    argv: list[str]
    scopes: list[str]
    tool_version: str | None

    def failure(self, reason: str) -> Measurement:
        """Retain the attempted command while refusing any numeric debt figure."""
        return Measurement(
            language=self.language,
            tool=self.tool,
            tool_version=self.tool_version,
            argv=self.argv,
            undocumented=None,
            files=None,
            scopes=self.scopes,
            not_measured_reason=reason,
        )


def measure_go(root: Path) -> Measurement:
    """Measure Go with the language's own parser.

    Go was recorded here as not measured, because no Go documentation *linter*
    is installed. ``go/parser`` is what those linters are built on, it ships
    with the toolchain, and it answers exactly the question the directive asks:
    which exported declarations carry no doc comment. That is a tool which
    understands the language, not a grep for a comment above an export.
    """
    scopes = ["*.go (tracked and nonignored untracked)"]
    argv = ["go", "run", GO_COVERAGE_TOOL]
    version = _version(["go", "version"], cwd=root)
    if version is None:
        return unmeasured(
            "go",
            "go/parser via tools/godoc_coverage",
            "the Go toolchain is not installed, so its parser could not be run",
            scopes,
        )
    try:
        qualified = qualify_repository(root)
    except GoMeasurementError as error:
        return unmeasured(
            "go",
            "go/parser via tools/godoc_coverage",
            f"the coverage tool produced no figure: {error}",
            scopes,
        )
    return Measurement(
        language="go",
        tool="go/parser via tools/godoc_coverage",
        tool_version=qualified.version,
        argv=argv,
        undocumented=qualified.undocumented,
        files=qualified.files,
        scopes=scopes,
        provenance=qualified.to_public_dict(),
    )


def measure_python(root: Path, scopes: Sequence[str]) -> Measurement:
    """Measure Python with ruff's pydocstyle rules for missing documentation.

    Only successful native reports containing ``D1xx`` findings are counted.
    Missing inputs, syntax errors, unavailable tools and empty native cohorts
    remain unmeasured. Style rules about existing docstrings are a separate
    quality obligation.
    """
    rules = "D100,D101,D102,D103,D104,D105,D106,D107,E902"
    interpreter = [sys.executable]
    if sys.flags.isolated:
        interpreter.append("-I")
    else:
        if sys.flags.ignore_environment:
            interpreter.append("-E")
        if sys.flags.no_user_site:
            interpreter.append("-s")
    argv = [
        *interpreter,
        "-m",
        "ruff",
        "check",
        "--select",
        rules,
        "--output-format",
        "json",
        *scopes,
    ]
    version = _version([*interpreter, "-m", "ruff", "--version"], cwd=root)
    attempt = _Attempt("python", "ruff (D1xx)", argv, list(scopes), version)
    if version is None:
        return Measurement(
            language="python",
            tool="ruff (D1xx)",
            tool_version=None,
            argv=argv,
            undocumented=None,
            files=None,
            scopes=list(scopes),
            not_measured_reason="ruff is not importable in this interpreter",
        )
    if not scopes:
        return attempt.failure("no Python scopes were specified")
    try:
        source_query = _run(
            [*interpreter, "-m", "ruff", "check", "--show-files", *scopes], cwd=root
        )
        completed = _run(argv, cwd=root)
    except (OSError, subprocess.SubprocessError) as error:
        return attempt.failure(f"Ruff could not complete: {error}")
    if source_query.returncode != 0 or not source_query.stdout.strip():
        return attempt.failure("Ruff found no complete Python source cohort")
    if completed.returncode not in (0, 1):
        return attempt.failure(f"Ruff exited {completed.returncode}: {completed.stderr.strip()}")
    try:
        report: list[dict[str, Any]] = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return attempt.failure("Ruff produced no JSON report")
    if not isinstance(report, list) or any(not isinstance(row, dict) for row in report):
        return attempt.failure("Ruff produced an invalid report shape")
    invalid = [row for row in report if row.get("code") not in rules.split(",")[:-1]]
    if invalid:
        return attempt.failure(
            f"Ruff refused an input: {invalid[0].get('code')}: {invalid[0].get('message')}"
        )
    undocumented = len(report)
    files = len({row["filename"] for row in report})
    return Measurement(
        language="python",
        tool="ruff (D1xx)",
        tool_version=version,
        argv=argv,
        undocumented=undocumented,
        files=files,
        scopes=list(scopes),
    )


def measure_typescript(root: Path, config: str) -> Measurement:
    """Measure TypeScript with ESLint, the tool that enforces the rule.

    The measurement runs the documentation rules over the whole frontend rather
    than over the enforced subset, so the figure is the debt outside the gate
    and not a restatement of the gate passing.
    """
    frontend = root / "studio" / "frontend"
    argv = ["npx", "eslint", ".", "--config", config, "-f", "json"]
    if not (frontend / config).is_file():
        return Measurement(
            language="typescript",
            tool="eslint + eslint-plugin-jsdoc",
            tool_version=None,
            argv=argv,
            undocumented=None,
            files=None,
            scopes=["studio/frontend"],
            not_measured_reason=f"measurement config {config} is not present",
        )
    version = _version(["npx", "eslint", "--version"], cwd=frontend)
    attempt = _Attempt(
        "typescript", "eslint + eslint-plugin-jsdoc", argv, ["studio/frontend"], version
    )
    if version is None:
        return attempt.failure("eslint could not report its version")
    try:
        completed = _run(argv, cwd=frontend)
    except (OSError, subprocess.SubprocessError) as error:
        return attempt.failure(f"eslint could not complete: {error}")
    if completed.returncode not in (0, 1):
        return attempt.failure(f"eslint exited {completed.returncode}: {completed.stderr.strip()}")
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return Measurement(
            language="typescript",
            tool="eslint + eslint-plugin-jsdoc",
            tool_version=version,
            argv=argv,
            undocumented=None,
            files=None,
            scopes=["studio/frontend"],
            not_measured_reason="eslint produced no JSON report",
        )
    if not isinstance(report, list) or any(not isinstance(row, dict) for row in report):
        return attempt.failure("eslint produced an invalid report shape")
    if not report:
        return attempt.failure("eslint measured no source files")
    if any(message.get("fatal", False) for row in report for message in row.get("messages", [])):
        return attempt.failure("eslint reported a fatal source error")
    undocumented, files = read_eslint_report(report)
    return Measurement(
        language="typescript",
        tool="eslint + eslint-plugin-jsdoc",
        tool_version=version,
        argv=argv,
        undocumented=undocumented,
        files=files,
        scopes=["studio/frontend"],
    )


def measure_rust(root: Path, manifest: str) -> Measurement:
    """Measure Rust with ``rustc``'s own ``missing_docs`` lint.

    This is the tool the directive names. It builds the crate, so it is run
    against a target directory of the caller's choosing to keep it out of a
    shared one.
    """
    try:
        measured = measure_rust_findings(root, manifest)
    except RustMeasurementError as error:
        return Measurement(
            language="rust",
            tool="rustc --force-warn missing_docs",
            tool_version=None,
            argv=[],
            undocumented=None,
            files=None,
            scopes=[manifest],
            not_measured_reason=str(error),
        )
    return Measurement(
        language="rust",
        tool="rustc --force-warn missing_docs",
        tool_version=measured.rustc_version,
        argv=list(measured.argv),
        undocumented=measured.undocumented,
        files=measured.files,
        scopes=list(measured.paths),
        provenance=measured.to_public_dict(),
    )


def unmeasured(language: str, tool: str, reason: str, scopes: Sequence[str]) -> Measurement:
    """Record a language whose measurement tool is not available.

    Stated rather than substituted: a figure from the wrong instrument reads as
    evidence and is worse than an honest gap.
    """
    return Measurement(
        language=language,
        tool=tool,
        tool_version=None,
        argv=[],
        undocumented=None,
        files=None,
        scopes=list(scopes),
        not_measured_reason=reason,
    )


def build_report(measurements: Sequence[Measurement], *, source_sha: str) -> dict[str, Any]:
    """Return the artefact recording every measurement and what produced it."""
    measured = [m for m in measurements if m.undocumented is not None]
    return {
        "languages": [m.to_public_dict() for m in measurements],
        "schema_version": DOCUMENTATION_DEBT_SCHEMA_VERSION,
        "source_sha256": source_sha,
        "summary": {
            "languages_measured": len(measured),
            "languages_not_measured": len(measurements) - len(measured),
            "undocumented_total": sum(m.undocumented or 0 for m in measured),
        },
    }


def main(argv: list[str] | None = None) -> int:
    """Measure every language and write the artefact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eslint-config", default="eslint.measure.js")
    parser.add_argument("--rust-manifest", default="engine/Cargo.toml")
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        help="a language to record as not measured in this run",
    )
    args = parser.parse_args(argv)
    root = args.repo.resolve()
    skip = set(args.skip)

    measurements: list[Measurement] = []
    if "python" in skip:
        measurements.append(
            unmeasured("python", "ruff (D1xx)", "skipped in this run", ["src", "tools", "tests"])
        )
    else:
        measurements.append(measure_python(root, ["src", "tools", "tests"]))
    if "typescript" in skip:
        measurements.append(
            unmeasured("typescript", "eslint", "skipped in this run", ["studio/frontend"])
        )
    else:
        measurements.append(measure_typescript(root, args.eslint_config))
    if "rust" in skip:
        measurements.append(
            unmeasured("rust", "rustc -W missing_docs", "skipped in this run", [args.rust_manifest])
        )
    else:
        measurements.append(measure_rust(root, args.rust_manifest))
    if "go" in skip:
        measurements.append(
            unmeasured(
                "go",
                "go/parser via tools/godoc_coverage",
                "skipped in this run",
                ["*.go (tracked)"],
            )
        )
    else:
        measurements.append(measure_go(root))
    measurements.append(
        unmeasured(
            "julia",
            "DocumenterTools / Aqua.jl",
            "no Julia documentation-coverage package is present in the shared julia "
            "environment, and adding one there would change an environment other "
            "seats' jobs run in",
            ["src/sc_neurocore/accel/julia"],
        )
    )
    measurements.append(
        unmeasured(
            "mojo",
            "mojo doc",
            "`mojo doc` emits a docstring JSON rather than a coverage figure, and it "
            "produced an empty document for a sampled kernel; a coverage method over "
            "its output has not been established, so no figure is claimed",
            ["src/sc_neurocore/accel/mojo"],
        )
    )

    sha = _run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()
    report = build_report(measurements, source_sha=sha)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    for measurement in measurements:
        if measurement.undocumented is None:
            print(f"  {measurement.language:<11} not measured: {measurement.not_measured_reason}")
        else:
            print(
                f"  {measurement.language:<11} {measurement.undocumented:>6} undocumented "
                f"in {measurement.files} files ({measurement.tool})"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
