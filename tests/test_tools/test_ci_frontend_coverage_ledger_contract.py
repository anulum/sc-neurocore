# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI contract of the frontend coverage ledger

"""Guard the frontend coverage ledger in the hosted frontend job and in the frontend package.

The ledger only protects anything while the job runs its checker after the
Vitest suite, against the committed ledger, with the measurement the checker
reads. These cases fail when any of those links is removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend"
UPLOAD_ARTIFACT = "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
LEDGER_SCHEMA = "sc-neurocore.studio-frontend-coverage-ledger.v1"


def _frontend_steps() -> list[dict[str, Any]]:
    """Return the parsed steps of the hosted frontend job."""
    workflow = cast(
        dict[str, Any],
        yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")),
    )
    return cast(list[dict[str, Any]], workflow["jobs"]["frontend"]["steps"])


def _step_index(steps: list[dict[str, Any]], name: str) -> int:
    """Return the index of the one step called ``name``."""
    matches = [index for index, step in enumerate(steps) if step.get("name") == name]
    assert len(matches) == 1, f"expected one step named {name!r}, found {len(matches)}"
    return matches[0]


def test_the_frontend_job_checks_the_ledger_after_installing_and_testing() -> None:
    """The ledger step runs in the frontend folder, after the install and the Vitest suite."""
    steps = _frontend_steps()
    ledger = _step_index(steps, "Frontend coverage ledger")
    assert _step_index(steps, "Install frontend dependencies") < ledger
    assert _step_index(steps, "Vitest") < ledger
    assert steps[ledger]["working-directory"] == "studio/frontend"
    assert steps[ledger]["run"] == "npm run test:coverage-ledger"
    assert "if" not in steps[ledger] and "continue-on-error" not in steps[ledger]


def test_the_frontend_job_keeps_the_measured_summary() -> None:
    """Every run keeps the summary it was judged on, so a first or disputed ledger can be read from it."""
    steps = _frontend_steps()
    upload = _step_index(steps, "Preserve the frontend coverage summary")
    assert upload == _step_index(steps, "Frontend coverage ledger") + 1
    step = steps[upload]
    assert step["if"] == "always()"
    assert step["uses"] == UPLOAD_ARTIFACT
    assert step["with"]["path"] == "studio/frontend/coverage/ledger/coverage-summary.json"
    assert step["with"]["if-no-files-found"] == "error"


def test_the_package_script_tests_the_checker_then_measures_then_checks() -> None:
    """The script runs the checker's tests at 100 %, the suite with coverage, then the check."""
    scripts = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["scripts"]
    parts = [part.strip() for part in scripts["test:coverage-ledger"].split("&&")]
    assert parts[0].startswith("node --test --experimental-test-coverage")
    for threshold in ("lines", "branches", "functions"):
        assert f"--test-coverage-{threshold}=100" in parts[0]
    assert parts[0].endswith("scripts/coverage-ledger.node-test.ts")
    assert parts[1] == "vitest run --coverage"
    assert parts[2] == (
        "node scripts/coverage-ledger.ts check coverage/ledger/coverage-summary.json "
        "coverage-ledger.json"
    )


def test_vitest_writes_the_summary_the_checker_reads() -> None:
    """The base Vitest configuration measures the source with v8 into the folder the script reads."""
    config = (FRONTEND / "vitest.config.ts").read_text(encoding="utf-8")
    for fragment in (
        'provider: "v8"',
        'include: ["src/**/*.{ts,tsx}"]',
        '"src/**/*.test.{ts,tsx}"',
        'reporter: ["text", "json-summary"]',
        'reportsDirectory: "coverage/ledger"',
    ):
        assert fragment in config, fragment


def test_the_committed_ledger_is_sorted_and_licensed() -> None:
    """The ledger exists with its schema, sorted files and two percentages each, beside its licence."""
    path = FRONTEND / "coverage-ledger.json"
    ledger = json.loads(path.read_text(encoding="utf-8"))
    assert ledger["schema"] == LEDGER_SCHEMA
    files = ledger["files"]
    assert list(files) == sorted(files)
    assert files, "the ledger names no file"
    for name, entry in files.items():
        assert name.startswith("src/"), name
        assert list(entry) == ["statements", "branches"], name
        assert all(0 <= value <= 100 for value in entry.values()), name
    assert (FRONTEND / "coverage-ledger.json.license").is_file()


def test_the_coverage_output_is_ignored() -> None:
    """The measurement is written to an ignored folder and never committed."""
    assert "studio/frontend/coverage/" in (ROOT / ".gitignore").read_text(encoding="utf-8").split()
