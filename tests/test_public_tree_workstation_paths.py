# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — tracked files name no workstation path

"""Tracked files name no workstation path, private record tree or monorepo location.

The repository is public. A tracked file, text or binary, must not carry an
absolute path of a development host, the name of the private monorepo, or a
location in the group's private record tree. The only exceptions are the
lines below, each with its reason, and each must keep matching so that a
removed exception is noticed.

The markers are assembled from parts, so this file does not match itself.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MARKERS = (
    "/" + "media/anulum",
    "/" + "home/anulum",
    "aaa_God_of_the_" + "Math_Collection",
    "." + "coordination/",
)

# Each allowed line, by file, with the reason it may name a marker.
ALLOWED: dict[str, tuple[tuple[str, str], ...]] = {
    ".gitignore": (
        (
            "." + "coordination/",
            "the ignore rule that keeps the private record tree out of the repository",
        ),
    ),
    "studio/frontend/src/analysisJob.test.ts": (
        (
            'message: "boom /' + 'home/anulum/secret",',
            "a redaction test feeds a host path to the redactor",
        ),
        (
            'expect(redacted.error).not.toContain("/' + 'home/anulum");',
            "the redaction test asserts its absence",
        ),
    ),
    "studio/frontend/src/modelScanJob.test.ts": (
        (
            'throw new Error("/' + 'home/anulum/secret leaked");',
            "a redaction test feeds a host path to the redactor",
        ),
        (
            'expect(pollFail.getState().error).not.toContain("/' + 'home/anulum");',
            "the redaction test asserts its absence",
        ),
    ),
    "tests/test_quantum_cognition_cli_memory_discipline.py": (
        (
            'assert "/' + 'media/anulum" not in source, module.__name__',
            "the test refuses host paths in sources",
        ),
        (
            'assert "/' + 'home/anulum" not in source, module.__name__',
            "the test refuses host paths in sources",
        ),
    ),
    "tools/snn_memory_discipline_audit.py": (
        (
            '"session": ".' + 'coordination/sessions/",',
            "the audit checks that a stimulus cites a group session",
        ),
        (
            '"handover": ".' + 'coordination/handovers/",',
            "the audit checks that a stimulus cites a group handover",
        ),
    ),
    "tests/test_tools/test_snn_memory_discipline_audit_validate_stimulus.py": (
        (
            '"session": ".' + 'coordination/sessions/SC-NEUROCORE/session.md",',
            "the audit test's valid stimulus",
        ),
        (
            '"handover": ".' + 'coordination/handovers/SC-NEUROCORE/handover.md",',
            "the audit test's valid stimulus",
        ),
    ),
}


def _marker_lines() -> list[tuple[str, str]]:
    """Return ``(path, line)`` for every tracked line, text or binary, that names a marker."""
    command = ["git", "grep", "-a", "-n", "--no-color", "-F"]
    for marker in MARKERS:
        command += ["-e", marker]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    found = []
    for row in result.stdout.splitlines():
        path, _number, line = row.split(":", 2)
        found.append((path, line.strip()))
    return found


def test_tracked_files_name_no_workstation_path() -> None:
    """Every tracked line naming a marker is an allowed, reasoned exception."""
    unexpected = [
        f"{path}: {line[:160]}"
        for path, line in _marker_lines()
        if line not in {allowed for allowed, _reason in ALLOWED.get(path, ())}
    ]
    assert unexpected == [], "workstation or private-record paths in tracked files:\n" + "\n".join(
        unexpected
    )


def test_every_exception_still_matches_a_tracked_line() -> None:
    """A removed exception is noticed: each allowed line is still present in its file."""
    present = set(_marker_lines())
    stale = [
        f"{path}: {line}"
        for path, entries in ALLOWED.items()
        for line, _ in entries
        if (path, line) not in present
    ]
    assert stale == []


def test_every_exception_states_a_reason() -> None:
    """Each allowed line carries a non-empty reason and names a marker."""
    for path, entries in ALLOWED.items():
        for line, reason in entries:
            assert reason.strip(), path
            assert any(marker in line for marker in MARKERS), (path, line)
