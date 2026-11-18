# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI contract that every live Studio browser spec runs

"""Guard that every live Studio browser spec is run by a hosted workflow, and the event suite with its environment.

Each ``studio/frontend/e2e/*-live.spec.ts`` needs a real backend, so the mocked
default Playwright run ignores it and a dedicated configuration selects it. Three
of them were selected by a configuration no workflow ran, or left out of the file
list of the step that ran it, so no hosted run had ever executed them. These cases
read every workflow's Playwright invocations, resolve each configuration's
``testMatch`` and ``testIgnore`` and the step's own file list. They fail when a
live spec runs under no configuration that starts its backend, when the mocked
default run selects one, and when the event suite runs without the fresh root and
the repository root on ``PYTHONPATH`` that its server needs.
"""

from __future__ import annotations

import fnmatch
import json
import re
import shlex
from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "studio" / "frontend"
E2E = FRONTEND / "e2e"
WORKFLOWS = ROOT / ".github" / "workflows"
DEFAULT_CONFIG = "playwright.config.ts"
EVENT_CONFIG = "playwright.event.config.ts"


def _strings(block: str) -> list[str]:
    """Return the double-quoted string literals of a TypeScript fragment."""
    return re.findall(r'"([^"]+)"', block)


def _patterns(text: str, key: str) -> list[str]:
    """Return the patterns a Playwright configuration gives under ``key``: one string or an array."""
    match = re.search(rf'{key}:\s*(\[.*?\]|"[^"]*")', text, flags=re.S)
    return [] if match is None else _strings(match.group(1))


def _matches(name: str, patterns: list[str]) -> bool:
    """Say whether a spec file name matches one of the patterns, which name files under the test folder."""
    return any(fnmatch.fnmatch(name, pattern.removeprefix("**/")) for pattern in patterns)


def selected_specs(config: str) -> set[str]:
    """Return the spec files a Playwright configuration selects in the test folder."""
    text = (FRONTEND / config).read_text(encoding="utf-8")
    include, ignore = _patterns(text, "testMatch"), _patterns(text, "testIgnore")
    return {
        path.name
        for path in E2E.glob("*.spec.ts")
        if _matches(path.name, include) and not _matches(path.name, ignore)
    }


def _scripts() -> dict[str, str]:
    """Return the frontend package's npm scripts."""
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    return cast(dict[str, str], package["scripts"])


def specs_of(command: str, depth: int = 0) -> set[tuple[str, str]]:
    """Return the specs one shell command runs through Playwright, with their configuration.

    Parameters
    ----------
    command : str
        One command, without line continuations.
    depth : int
        How many npm scripts deep the expansion is; a cycle stops at 8.

    Returns
    -------
    set of (str, str)
        Pairs of the configuration and a spec file it runs: the configuration's
        selection, narrowed to the file names the command lists when it lists any.
        npm scripts are followed.
    """
    # Only a Playwright or npm command is read; other script lines (Python, comments) need no shell parse.
    if depth > 8 or ("playwright" not in command and not command.startswith("npm run ")):
        return set()
    words = shlex.split(command, comments=False, posix=True)
    if words[:2] == ["npm", "run"] and len(words) > 2:
        script = _scripts().get(words[2], "")
        return {
            pair for part in re.split(r"&&|;", script) for pair in specs_of(part.strip(), depth + 1)
        }
    if "playwright" not in words or "test" not in words:
        return set()
    config = DEFAULT_CONFIG
    for index, word in enumerate(words):
        if word == "--config" and index + 1 < len(words):
            config = words[index + 1]
        elif word.startswith("--config="):
            config = word.split("=", 1)[1]
    listed = {word for word in words if word.endswith(".spec.ts")}
    selected = selected_specs(config)
    return {(config, spec) for spec in (selected & listed if listed else selected)}


def _commands(script: str) -> list[str]:
    """Split a run script into single commands, joining continuation lines."""
    joined = re.sub(r"\\\n\s*", " ", script)
    return [
        part.strip()
        for line in joined.splitlines()
        for part in re.split(r"&&|;", line)
        if part.strip()
    ]


def _steps() -> list[tuple[str, list[dict[str, Any]], int]]:
    """Return every workflow step as (workflow and job, the job's steps, the step's index)."""
    found: list[tuple[str, list[dict[str, Any]], int]] = []
    for workflow in sorted(WORKFLOWS.glob("*.yml")):
        data = cast(dict[str, Any], yaml.safe_load(workflow.read_text(encoding="utf-8")))
        for job_name, job in cast(dict[str, Any], data.get("jobs", {})).items():
            steps = cast(list[dict[str, Any]], job.get("steps", []))
            found.extend(
                (f"{workflow.name}:{job_name}", steps, index) for index in range(len(steps))
            )
    return found


def run_specs() -> set[tuple[str, str]]:
    """Return every (configuration, spec file) pair some hosted workflow step runs."""
    pairs: set[tuple[str, str]] = set()
    for _where, steps, index in _steps():
        for command in _commands(str(steps[index].get("run", ""))):
            pairs |= specs_of(command)
    return pairs


def _live_specs() -> set[str]:
    """Return the live spec files of the test folder."""
    return {path.name for path in E2E.glob("*-live.spec.ts")}


def test_the_reading_resolves_configurations_and_file_lists() -> None:
    """A configuration's selection is read, and a step's file list narrows it.

    The checks below would pass for nothing if the reading found no spec; these
    pin that it reads the real configurations and honours a listed subset.
    """
    assert "hardware-catalogue-live.spec.ts" in selected_specs("playwright.live.config.ts")
    assert "fit-live.spec.ts" in selected_specs("playwright.graph.config.ts")
    narrowed = specs_of(
        "npx playwright test --config playwright.live.config.ts equation-ir-live.spec.ts"
    )
    assert narrowed == {("playwright.live.config.ts", "equation-ir-live.spec.ts")}


def test_the_mocked_default_run_ignores_every_live_spec() -> None:
    """A live spec needs a real backend, so the mocked default configuration must not select it."""
    live = _live_specs()
    assert live, "no live spec found"
    selected = sorted(live & selected_specs(DEFAULT_CONFIG))
    assert not selected, f"live specs the mocked default run selects: {selected}"


def test_every_live_spec_runs_in_a_hosted_workflow() -> None:
    """Each live spec runs in some workflow step under a configuration that starts its backend."""
    run = {spec for config, spec in run_specs() if config != DEFAULT_CONFIG}
    missing = sorted(_live_specs() - run)
    assert not missing, f"live specs no hosted workflow runs against a backend: {missing}"


def test_the_event_suite_runs_with_its_root_and_the_repository_on_pythonpath() -> None:
    """The event suite's step gives it a fresh root and the repository root, after the bundle build.

    ``e2e/event_training_server.py`` imports ``tests.event_dataset_support`` from the
    repository root and refuses a root that is not absolute, and the configuration
    previews the built bundle.
    """
    event_steps = [
        (where, steps, index)
        for where, steps, index in _steps()
        if any(EVENT_CONFIG in command for command in _commands(str(steps[index].get("run", ""))))
    ]
    assert len(event_steps) == 1, [where for where, _steps, _index in event_steps]
    _where, steps, index = event_steps[0]
    script = str(steps[index]["run"])
    assert re.search(r'export SC_NEUROCORE_STUDIO_EVENT_TEST_ROOT="\$\w+/', script)
    assert re.search(r'export PYTHONPATH="\$GITHUB_WORKSPACE', script)
    builds = [
        position
        for position, step in enumerate(steps[:index])
        if "npm run build" in str(step.get("run", ""))
        and step.get("if") in (None, steps[index].get("if"))
    ]
    assert builds, "no earlier step on the same leg builds the bundle the event suite previews"
