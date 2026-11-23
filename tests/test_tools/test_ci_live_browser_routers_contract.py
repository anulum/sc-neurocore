# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — CI contract of the routers the live browser journeys need

"""Guard the router of every FPGA target the live browser journeys route, in the hosted test job.

The catalogue-to-silicon journeys choose the Studio's ECP5 target and click
"Synthesise + Route". The leg that runs them installed only the iCE40 router,
so the first journey to reach that click stopped at place and route ("nextpnr-ecp5
not found") the first time the live suite got that far. These cases fail when a
live journey routes a target whose router and chip database no earlier step of
that leg installs at the toolchain manifest's version, and when the reading of the
journeys finds no routed target at all.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast


from sc_neurocore.studio.synthesis_profiles import synthesis_target_profiles
from tools.toolchain_manifest import MANIFEST, load_manifest
from tools.ci_workflow_inventory import ci_job

ROOT = Path(__file__).resolve().parents[2]
E2E = ROOT / "studio" / "frontend" / "e2e"

#: A journey routes when it clicks the synthesis panel's place-and-route button.
ROUTE_BUTTON = '"Synthesise + Route"'

#: A journey chooses a target in the synthesis panel's list by the target's key.
TARGET_CHOICE = re.compile(r'selectOption\("([a-z0-9]+)"\)')


def _test_steps() -> list[dict[str, Any]]:
    """Return the parsed steps of the hosted test job."""
    return cast(list[dict[str, Any]], ci_job("test")["steps"])


def _live_step(steps: list[dict[str, Any]]) -> int:
    """Return the index of the one step that runs the live Playwright configuration."""
    matches = [
        index
        for index, step in enumerate(steps)
        if "playwright.live.config.ts" in str(step.get("run", ""))
    ]
    assert len(matches) == 1, matches
    return matches[0]


def _live_specs(run: str) -> list[str]:
    """Return the spec files a run script names."""
    return re.findall(r"[\w-]+-live\.spec\.ts", run)


def _routed_targets(specs: list[str]) -> set[str]:
    """Return the registered targets that a named journey chooses and then routes."""
    keys = {profile.target for profile in synthesis_target_profiles()}
    targets: set[str] = set()
    for name in specs:
        text = (E2E / name).read_text(encoding="utf-8")
        if ROUTE_BUTTON in text:
            targets |= {key for key in TARGET_CHOICE.findall(text) if key in keys}
    return targets


def _installs_before(steps: list[dict[str, Any]], index: int) -> str:
    """Return the run scripts of the steps before ``index`` that run on every leg it runs on."""
    condition = steps[index].get("if")
    return "\n".join(
        str(step.get("run", "")) for step in steps[:index] if step.get("if") in (None, condition)
    )


def test_the_reading_finds_the_target_the_catalogue_journeys_route() -> None:
    """The live step runs the catalogue-to-silicon journeys, and they route the ECP5 target.

    A reading that found no routed target would let the next case pass for nothing.
    """
    steps = _test_steps()
    specs = _live_specs(str(steps[_live_step(steps)]["run"]))
    assert "catalogue-to-silicon-live.spec.ts" in specs
    assert "ecp5" in _routed_targets(specs)


def test_every_routed_target_has_its_router_installed_before_the_live_step() -> None:
    """Each target a live journey routes has its router and chip database installed first.

    The install runs on every leg that runs the live step (with no condition, or
    with the live step's own), before it, at the version the toolchain manifest pins.
    """
    steps = _test_steps()
    index = _live_step(steps)
    installed = _installs_before(steps, index)
    apt = load_manifest(ROOT / MANIFEST).apt
    routers = {profile.target: profile.routing_tool for profile in synthesis_target_profiles()}
    for target in sorted(_routed_targets(_live_specs(str(steps[index]["run"])))):
        router = routers[target]
        assert router is not None, f"a live journey routes {target}, which has no router"
        for package in (router, f"{router}-chipdb"):
            assert package in apt, f"the toolchain manifest does not pin {package}"
            assert f"{package}={apt[package]}" in installed, (
                f"no step before the live journeys installs {package}={apt[package]}"
            )
