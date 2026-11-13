# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI contract of the frontend lint

"""Guard the frontend lint in the hosted frontend job, in the package script and in the local preflight.

Before the job ran it, only the local preflight linted the frontend, so a lint
regression pushed without the hook reached main unseen. These cases fail when
the job loses the step, runs it after the tests or lets it fail quietly, when
the script stops refusing warnings, and when the preflight and the job stop
running the same script.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import yaml

import tools.preflight

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend"


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


def test_the_frontend_job_lints_after_installing_and_before_the_tests() -> None:
    """The lint step runs the package script in the frontend folder, between the install and Vitest."""
    steps = _frontend_steps()
    lint = _step_index(steps, "Frontend lint")
    assert _step_index(steps, "Install frontend dependencies") < lint < _step_index(steps, "Vitest")
    assert steps[lint]["working-directory"] == "studio/frontend"
    assert steps[lint]["run"] == "npm run lint"
    assert "if" not in steps[lint] and "continue-on-error" not in steps[lint]


def test_the_lint_script_lints_the_whole_package_and_refuses_warnings() -> None:
    """``npm run lint`` covers the whole package and fails on a warning as on an error."""
    scripts = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert scripts["lint"] == "eslint . --max-warnings=0"


def test_the_local_preflight_and_the_job_run_the_same_lint() -> None:
    """The preflight's first frontend check is the script the hosted job runs."""
    assert tools.preflight.FRONTEND_CHECKS[0] == ["npm", "run", "lint"]
    steps = _frontend_steps()
    preflight_lint = " ".join(tools.preflight.FRONTEND_CHECKS[0])
    assert steps[_step_index(steps, "Frontend lint")]["run"] == preflight_lint
