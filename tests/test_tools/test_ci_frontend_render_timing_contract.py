# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI contract of the Studio render timing measurement

"""Guard how the Studio's render timing measurement is run and kept.

The measurement times the built bundle against a real backend, only when the
workflow is started by hand, and keeps its artefact. These cases fail when the
hosted step, its artefact, the package script or the configuration drift from
that.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend"
UPLOAD_ARTIFACT = "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
BY_HAND = "github.event_name == 'workflow_dispatch'"
ARTEFACT = "${{ runner.temp }}/render-timing/render-timing.json"


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


def test_the_measurement_runs_by_hand_after_the_build_and_the_backend() -> None:
    """The step runs the timing configuration on a hand-started run, once both servers can start."""
    steps = _frontend_steps()
    measure = _step_index(steps, "Studio render timing measurement")
    assert _step_index(steps, "TypeScript and Vite build") < measure
    assert _step_index(steps, "Install Playwright Chromium") < measure
    assert _step_index(steps, "Install live Studio backend dependencies") < measure
    step = steps[measure]
    assert step["if"] == BY_HAND
    assert step["working-directory"] == "studio/frontend"
    assert step["run"] == "npx playwright test --config playwright.timing.config.ts"
    assert step["env"] == {"SC_NEUROCORE_RENDER_TIMING_OUT": ARTEFACT}


def test_the_training_workload_has_its_backend_before_the_measurement() -> None:
    """The training workload needs PyTorch, which the live backend's locks do not hold.

    The CPU build is installed from its hash-locked file right before the
    measurement, on the same hand-started runs only; without it the training
    run fails and the whole measurement with it.
    """
    steps = _frontend_steps()
    torch = _step_index(steps, "Install PyTorch for the training workload")
    assert _step_index(steps, "Install live Studio backend dependencies") < torch
    assert torch + 1 == _step_index(steps, "Studio render timing measurement")
    step = steps[torch]
    assert step["if"] == BY_HAND
    assert step["run"].split() == [
        "python",
        "-m",
        "pip",
        "install",
        "--require-hashes",
        "--only-binary",
        ":all:",
        "-r",
        "requirements/ci-torch-cpu.txt",
    ]


def test_the_artefact_is_kept_from_a_hand_started_run() -> None:
    """The artefact the step writes is uploaded right after it, and its absence is an error."""
    steps = _frontend_steps()
    upload = _step_index(steps, "Preserve the render timing artefact")
    assert upload == _step_index(steps, "Studio render timing measurement") + 1
    step = steps[upload]
    assert step["if"] == BY_HAND
    assert step["uses"] == UPLOAD_ARTIFACT
    assert step["with"]["name"] == "studio-render-timing"
    assert step["with"]["path"] == ARTEFACT
    assert step["with"]["if-no-files-found"] == "error"


def test_the_package_script_builds_before_it_measures() -> None:
    """Locally, the script builds the bundle first, as the hosted job does."""
    scripts = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))["scripts"]
    assert (
        scripts["test:e2e:timing"]
        == "npm run build && playwright test --config playwright.timing.config.ts"
    )
    assert "npm run test:render-timing-report" in scripts["test"]


def test_the_configuration_times_the_built_bundle_and_nothing_else() -> None:
    """The timing configuration serves the build with ``vite preview`` and runs only the timing spec."""
    config = (FRONTEND / "playwright.timing.config.ts").read_text(encoding="utf-8")
    assert re.search(r'testMatch: \["render-timing-live\.spec\.ts"\]', config)
    assert "npm run preview --" in config
    assert "npm run dev" not in config
    assert "sc_neurocore.studio.app:create_app" in config
    default = (FRONTEND / "playwright.config.ts").read_text(encoding="utf-8")
    assert '"render-timing-live.spec.ts",' in default
