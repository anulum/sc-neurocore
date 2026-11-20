# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Playwright browsers in namespaced CI steps

"""Every CI step that moves the cache directory and runs Playwright installs its browser there first.

Playwright looks for its browsers under ``XDG_CACHE_HOME`` unless ``PLAYWRIGHT_BROWSERS_PATH`` says otherwise.
A step that exports its own cache directory and then runs ``npx playwright test`` without installing finds no
browser and fails at launch, as the event training browser suite did.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import cast

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))
_TEST = re.compile(r"\bnpx playwright test\b")
_INSTALL = re.compile(r"\bnpx playwright install\b")
_OWN_CACHE = re.compile(r"\bXDG_CACHE_HOME=")


def run_steps(workflow: Path) -> list[tuple[str, str]]:
    """Return each step's name and ``run`` script in one workflow.

    Parameters
    ----------
    workflow : Path
        The workflow file.

    Returns
    -------
    list of tuple
        The step name (or its index) and its script, for every step with a ``run`` script.
    """
    document = cast(dict[str, object], yaml.safe_load(workflow.read_text(encoding="utf-8")))
    jobs = cast(dict[str, dict[str, object]], document.get("jobs") or {})
    found: list[tuple[str, str]] = []
    for job_name, job in jobs.items():
        for index, step in enumerate(cast(list[dict[str, object]], job.get("steps") or [])):
            script = step.get("run")
            if isinstance(script, str):
                found.append((f"{job_name}: {step.get('name', index)}", script))
    return found


def missing_installs(script: str) -> bool:
    """Return whether a script moves the cache and tests with Playwright without installing a browser.

    Parameters
    ----------
    script : str
        A step's ``run`` script.

    Returns
    -------
    bool
        True when the script exports ``XDG_CACHE_HOME`` and runs ``npx playwright test``, but neither installs a
        browser nor sets ``PLAYWRIGHT_BROWSERS_PATH`` before the first test command.
    """
    if not (_OWN_CACHE.search(script) and _TEST.search(script)):
        return False
    first_test = _TEST.search(script)
    assert first_test is not None
    before = script[: first_test.start()]
    return not (_INSTALL.search(before) or "PLAYWRIGHT_BROWSERS_PATH" in before)


@pytest.mark.parametrize(
    ("script", "expected"),
    [
        ('export XDG_CACHE_HOME="$n/cache"\nnpx playwright test --config a.ts\n', True),
        (
            'export XDG_CACHE_HOME="$n/cache"\nnpx playwright install --with-deps chromium\nnpx playwright test\n',
            False,
        ),
        (
            'export XDG_CACHE_HOME="$n/cache"\nexport PLAYWRIGHT_BROWSERS_PATH="$b"\nnpx playwright test\n',
            False,
        ),
        ("npx playwright test --config a.ts\n", False),
        ('export XDG_CACHE_HOME="$n/cache"\npython -m pytest\n', False),
        (
            'export XDG_CACHE_HOME="$n/cache"\nnpx playwright test\nnpx playwright install chromium\n',
            True,
        ),
    ],
    ids=[
        "no-install",
        "install",
        "browsers-path",
        "default-cache",
        "no-playwright",
        "install-after-test",
    ],
)
def test_the_rule_reads_one_script(script: str, expected: bool) -> None:
    """The rule flags a moved cache with no browser installed before the first test command."""
    assert missing_installs(script) is expected


def test_every_namespaced_playwright_step_installs_its_browser() -> None:
    """No step of any workflow moves the cache and runs Playwright without a browser in it."""
    assert WORKFLOWS
    offenders = [
        f"{workflow.name} / {step}"
        for workflow in WORKFLOWS
        for step, script in run_steps(workflow)
        if missing_installs(script)
    ]
    assert offenders == []
