# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Module-level tests from former test_region.py

"""Module-level tests from former test_region.py."""

from __future__ import annotations

from region_support import *  # noqa: F403

def test_selector_ignores_busy_regions_and_chooses_smallest_fit() -> None:
    busy = _region(0, neurons=512)
    busy.state = RegionState.ALLOCATED
    regions = {
        0: busy,
        1: _region(1, neurons=2048),
        2: _region(2, neurons=1024),
    }

    assert select_region_multi_die(regions, 512) == 2
def test_health_score_clamps_at_zero() -> None:
    health = RegionHealth(region_id=0, error_count=20, temperature_c=200.0, age_hours=200_000)

    assert health.health_score == 0.0
    assert health.is_degraded
def test_historical_surface_reexports_owner_objects_without_wrappers() -> None:
    assert compatibility_surface.RegionState is region_owner.RegionState
    assert compatibility_surface.HWRegion is region_owner.HWRegion
    assert compatibility_surface.RegionHealth is region_owner.RegionHealth
    assert compatibility_surface.select_region_multi_die is region_owner.select_region_multi_die
def test_region_definitions_have_one_owner() -> None:
    facade_tree = ast.parse(Path(compatibility_surface.__file__).read_text(encoding="utf-8"))
    owner_tree = ast.parse(Path(region_owner.__file__).read_text(encoding="utf-8"))

    facade_definitions = {
        node.name for node in facade_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    owner_definitions = {
        node.name for node in owner_tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    owned_names = {"RegionState", "HWRegion", "select_region_multi_die", "RegionHealth"}

    assert facade_definitions.isdisjoint(owned_names)
    assert owner_definitions == owned_names
