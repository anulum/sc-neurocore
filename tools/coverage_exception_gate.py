# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Coverage union gate with an owner-approved platform exception register

"""Hold the coverage union to 100 % except the lines an owner decision registers.

The coverage union of the main CI workflow is complete only on a host that reaches every line.
A line that runs only on a physical GPU cannot run on a hosted runner, so the owner allowed a
narrow platform exception backed by physical-GPU evidence. This gate reads the union's
``coverage json`` report, the register ``tools/coverage_platform_exceptions.json`` and the
evidence record it names, and fails when:

- a line is missing from the union and the register does not name it;
- a registered line's source text no longer has the sha256 the register records;
- the report has any missing branch, which the register cannot except;
- a source of the excepted path differs from the digest in the evidence record, so the
  physical-GPU evidence no longer describes the code;
- the evidence record does not cite the register's decisions, or does not show every
  registered line run on a physical GPU.

Usage::

    python tools/coverage_exception_gate.py --coverage-json coverage.json

Exit status: 0 when the union passes, 1 with findings, 2 when an input is missing or malformed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

#: The register of the owner-approved platform exceptions, below the checkout.
REGISTER = "tools/coverage_platform_exceptions.json"

#: The schema the register declares.
REGISTER_SCHEMA = "sc_neurocore.coverage_platform_exceptions.v1"

#: The schema the physical-GPU evidence record declares.
EVIDENCE_SCHEMA = "sc_neurocore.gpu_waveform_encoder_evidence.v1"


class GateInputError(ValueError):
    """An input of the gate is missing or malformed."""


@dataclass(frozen=True)
class RegisteredLine:
    """One registered line.

    Attributes
    ----------
    path : str
        The source file, relative to the checkout.
    line : int
        The 1-based line number.
    source_sha256 : str
        The sha256 of the line's text without its line terminator.
    """

    path: str
    line: int
    source_sha256: str


def _load_json(path: Path, what: str) -> dict[str, Any]:
    """Return a JSON object read from ``path``, or refuse it naming ``what``."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise GateInputError(f"{what} {path} cannot be read: {error}") from error
    if not isinstance(data, dict):
        raise GateInputError(f"{what} {path} is not a JSON object")
    return data


def line_sha256(root: Path, path: str, line: int) -> str | None:
    """Return the sha256 of one source line's text, or None when the line does not exist.

    Parameters
    ----------
    root : Path
        The checkout.
    path : str
        The file, relative to ``root``.
    line : int
        The 1-based line number.

    Returns
    -------
    str or None
        The hex digest of the line's UTF-8 text without its terminator.
    """
    try:
        lines = (root / path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    if not 1 <= line <= len(lines):
        return None
    return hashlib.sha256(lines[line - 1].encode("utf-8")).hexdigest()


def file_sha256(root: Path, path: str) -> str | None:
    """Return the sha256 of a file's bytes, or None when it cannot be read.

    Parameters
    ----------
    root : Path
        The checkout.
    path : str
        The file, relative to ``root``.

    Returns
    -------
    str or None
        The hex digest.
    """
    try:
        return hashlib.sha256((root / path).read_bytes()).hexdigest()
    except OSError:
        return None


def load_register(root: Path) -> tuple[dict[str, Any], list[RegisteredLine]]:
    """Return the register and its entries, or refuse a malformed one.

    Parameters
    ----------
    root : Path
        The checkout.

    Returns
    -------
    tuple
        The register object and its entries.

    Raises
    ------
    GateInputError
        If the register is missing, declares another schema, or holds a malformed entry.
    """
    register = _load_json(root / REGISTER, "register")
    if register.get("schema") != REGISTER_SCHEMA:
        raise GateInputError(
            f"register declares {register.get('schema')!r}, not {REGISTER_SCHEMA!r}"
        )
    entries: list[RegisteredLine] = []
    for raw in register.get("entries", []):
        try:
            entries.append(
                RegisteredLine(str(raw["path"]), int(raw["line"]), str(raw["source_sha256"]))
            )
        except (KeyError, TypeError, ValueError) as error:
            raise GateInputError(f"register entry {raw!r} is malformed: {error}") from error
    for key in ("decisions", "evidence", "source_set"):
        if not register.get(key):
            raise GateInputError(f"register lacks {key!r}")
    return register, entries


def evaluate(root: Path, report: dict[str, Any]) -> tuple[list[str], int, int, int]:
    """Return the findings, the statement count, the missing count and the excepted count.

    Parameters
    ----------
    root : Path
        The checkout.
    report : dict
        The union's ``coverage json`` report.

    Returns
    -------
    tuple
        The findings (empty when the union passes), the statements, the missing statements and
        the missing statements the register excepts.

    Raises
    ------
    GateInputError
        If the report, the register or the evidence record is malformed.
    """
    register, entries = load_register(root)
    files = report.get("files")
    if not isinstance(files, dict) or not files:
        raise GateInputError("coverage report has no measured files")
    registered = {(entry.path, entry.line): entry for entry in entries}
    findings: list[str] = []
    statements = missing_total = excepted = 0
    for path, data in sorted(files.items()):
        summary = data.get("summary", {})
        statements += int(summary.get("num_statements", 0))
        missing = [int(line) for line in data.get("missing_lines", [])]
        missing_total += len(missing)
        for line in missing:
            entry = registered.get((path, line))
            if entry is None:
                findings.append(f"{path}:{line} is not covered and not in the register")
            else:
                excepted += 1
        for branch in data.get("missing_branches", []):
            findings.append(f"{path}: branch {branch} is not covered; branches cannot be excepted")
    for entry in entries:
        actual = line_sha256(root, entry.path, entry.line)
        if actual != entry.source_sha256:
            findings.append(
                f"{entry.path}:{entry.line} changed since it was registered "
                f"(sha256 {actual}, registered {entry.source_sha256})"
            )
    findings.extend(_evidence_findings(root, register, entries))
    return findings, statements, missing_total, excepted


def _evidence_findings(
    root: Path, register: dict[str, Any], entries: list[RegisteredLine]
) -> list[str]:
    """Return what the physical-GPU evidence record fails to prove about the registered lines."""
    evidence = _load_json(root / str(register["evidence"]), "evidence record")
    findings: list[str] = []
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        findings.append(
            f"evidence record declares {evidence.get('schema')!r}, not {EVIDENCE_SCHEMA!r}"
        )
    if evidence.get("decisions") != register["decisions"]:
        findings.append("evidence record does not cite the register's decisions")
    recorded = evidence.get("source_sha256", {})
    for path in register["source_set"]:
        actual = file_sha256(root, path)
        if actual is None:
            findings.append(f"{path} of the excepted path's sources cannot be read")
        elif recorded.get(path) != actual:
            findings.append(f"{path} differs from the physical-GPU evidence; record it again")
    covered = {
        (item["path"], int(item["line"])) for item in evidence.get("excepted_lines_covered", [])
    }
    for entry in entries:
        if (entry.path, entry.line) not in covered:
            findings.append(f"{entry.path}:{entry.line} is not shown run on a physical GPU")
    return findings


def main(argv: list[str] | None = None) -> int:
    """Run the gate on a coverage report and print its verdict.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments; the process's own when None.

    Returns
    -------
    int
        0 when the union passes, 1 with findings, 2 for a missing or malformed input.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--coverage-json", required=True, type=Path, help="the union's coverage json report"
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="the checkout")
    args = parser.parse_args(argv)
    try:
        report = _load_json(args.coverage_json, "coverage report")
        findings, statements, missing, excepted = evaluate(args.root, report)
    except GateInputError as error:
        print(f"coverage exception gate: {error}", file=sys.stderr)
        return 2
    for finding in findings:
        print(f"FINDING {finding}")
    covered = statements - missing
    print(
        f"coverage exception gate: {covered} of {statements} statements covered; "
        f"{excepted} of {missing} missing excepted by the register; "
        f"{'pass' if not findings else f'{len(findings)} finding(s)'}"
    )
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
