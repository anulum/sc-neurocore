# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Requirement sets installed together pin the same versions

"""Hash-pinned sets installed in one step must agree on every shared package.

Each set is compiled on its own, so two sets can pin one transitive package to
different versions. pip then refuses the combined install, which surfaces only
when a workflow runs. Every workflow step that installs several sets with
``--require-hashes`` is found and checked here, so a recompiled set that
disagrees with a neighbour fails locally.
"""

from __future__ import annotations

import re
from itertools import combinations
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[2]
_SET = re.compile(r"requirements/[A-Za-z0-9_.-]+\.txt")
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s\\;]+)(\s*;[^\\]*)?", re.MULTILINE)


def _run_blocks() -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    for workflow in sorted((REPO / ".github" / "workflows").glob("*.yml")):
        document: Any = yaml.safe_load(workflow.read_text(encoding="utf-8"))
        for job in (document.get("jobs") or {}).values():
            for step in job.get("steps") or []:
                run = step.get("run") if isinstance(step, dict) else None
                if isinstance(run, str) and "--require-hashes" in run:
                    blocks.append((workflow.name, run))
    return blocks


def _pins(path: Path) -> dict[str, str]:
    """Unconditional pins; a pin under an environment marker is left to its marker."""
    return {
        name.lower().replace("_", "-"): version
        for name, version, marker in _PIN.findall(path.read_text(encoding="utf-8"))
        if not marker
    }


def _co_installed() -> list[tuple[str, tuple[str, ...]]]:
    installs: list[tuple[str, tuple[str, ...]]] = []
    for workflow, run in _run_blocks():
        # A command continued with a trailing backslash is one command.
        for command in run.replace("\\\n", " ").splitlines():
            if "--require-hashes" not in command:
                continue
            sets = tuple(dict.fromkeys(_SET.findall(command)))
            if len(sets) > 1:
                installs.append((workflow, sets))
    return installs


def test_workflows_install_several_sets_together() -> None:
    """The check below has something to check: several multi-set installs exist."""
    installs = _co_installed()
    assert len(installs) >= 2
    assert any("requirements/hub.txt" in sets for _workflow, sets in installs)


def test_sets_installed_together_pin_shared_packages_alike() -> None:
    conflicts: list[str] = []
    for workflow, sets in _co_installed():
        pins = {name: _pins(REPO / name) for name in sets}
        for left, right in combinations(sets, 2):
            for package in sorted(pins[left].keys() & pins[right].keys()):
                if pins[left][package] != pins[right][package]:
                    conflicts.append(
                        f"{workflow}: {package} is {pins[left][package]} in {left} "
                        f"and {pins[right][package]} in {right}"
                    )
    assert conflicts == []
