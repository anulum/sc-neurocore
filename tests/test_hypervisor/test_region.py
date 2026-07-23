# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypervisor hardware-region contracts

"""Verify physical region geometry, placement, health, and ownership."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from sc_neurocore.hypervisor import hypervisor as compatibility_surface
from sc_neurocore.hypervisor import region as region_owner
from sc_neurocore.hypervisor.region import (
    HWRegion,
    RegionHealth,
    RegionState,
    select_region_multi_die,
)


def _region(rid: int = 0, neurons: int = 1024, base: int = 0x4000_0000) -> HWRegion:
    return HWRegion(
        region_id=rid,
        num_neurons=neurons,
        num_synapses=neurons * 16,
        axi_base_addr=base,
        axi_size=0x1000,
        die_id=0,
    )


class TestHWRegion:
    def test_defaults(self) -> None:
        r = _region()
        assert r.is_free
        assert r.state == RegionState.FREE

    def test_axi_end_addr(self) -> None:
        r = _region(base=0x1000)
        assert r.axi_end_addr == 0x1000 + 0x1000

    def test_contains_addr(self) -> None:
        r = _region(base=0x1000)
        assert r.contains_addr(0x1000) is True
        assert r.contains_addr(0x1FFF) is True
        assert r.contains_addr(0x2000) is False
        assert r.contains_addr(0x0FFF) is False


class TestMultiDieSelector:
    def test_prefers_die(self) -> None:
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
            1: HWRegion(1, 1024, 0, 0x2000, 0x1000, die_id=1),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=1)
        assert rid == 1

    def test_fallback_any_die(self) -> None:
        regions = {
            0: HWRegion(0, 1024, 0, 0x1000, 0x1000, die_id=0),
        }
        rid = select_region_multi_die(regions, 512, preferred_die=5)
        assert rid == 0

    def test_no_fit(self) -> None:
        regions = {
            0: HWRegion(0, 64, 0, 0x1000, 0x1000, die_id=0),
        }
        assert select_region_multi_die(regions, 512) is None


class TestRegionHealth:
    def test_healthy_default(self) -> None:
        rh = RegionHealth(region_id=0)
        assert rh.health_score == pytest.approx(1.0)
        assert not rh.is_degraded

    def test_degraded_by_errors(self) -> None:
        rh = RegionHealth(region_id=0, error_count=5)
        assert rh.health_score < 0.8
        assert rh.is_degraded

    def test_temperature_penalty(self) -> None:
        rh = RegionHealth(region_id=0, temperature_c=100.0)
        assert rh.health_score < 1.0

    def test_record_error(self) -> None:
        rh = RegionHealth(region_id=0)
        rh.record_error()
        assert rh.error_count == 1


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
