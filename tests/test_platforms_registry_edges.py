# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for hardware-profile derived properties and registry edges

"""Contracts for HardwareProfile derived Q-format properties and registry lookup edges."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler.platforms import registry
from sc_neurocore.compiler.platforms.registry import (
    HardwareProfile,
    discover_platforms,
    get_profile,
    list_profiles,
    load_toml_profiles_dir,
    register_platform_hook,
)


def test_signed_profile_q_format_properties() -> None:
    """A signed Q9.9 profile reports its integer bits, label, range and resolution."""
    profile = HardwareProfile(
        name="signed_demo",
        vendor="Demo",
        family="Auto",
        platform_class="fpga",
        data_width=18,
        fraction=9,
    )

    assert profile.int_bits == 8
    assert profile.q_format_label == "Q8.9"
    assert profile.max_value == pytest.approx(((1 << 17) - 1) / (1 << 9))
    assert profile.min_value == pytest.approx(-(1 << 17) / (1 << 9))
    assert profile.resolution == pytest.approx(1.0 / (1 << 9))


def test_unsigned_profile_q_format_properties() -> None:
    """An unsigned UQ profile keeps the sign bit and reports a zero minimum value."""
    profile = HardwareProfile(
        name="unsigned_demo",
        vendor="Demo",
        family="Auto",
        platform_class="fpga",
        data_width=16,
        fraction=8,
        signed=False,
    )

    assert profile.int_bits == 8
    assert profile.q_format_label == "UQ8.8"
    assert profile.max_value == pytest.approx(((1 << 16) - 1) / (1 << 8))
    assert profile.min_value == 0.0


def test_from_constraints_mid_power_budget_selects_double_precision() -> None:
    """A mid-range power budget (10–100 mW) auto-selects double the precision width."""
    profile = HardwareProfile.from_constraints(
        "mid_power_chip",
        max_power_budget_mw=50.0,
        min_precision_bits=10,
    )

    assert profile.data_width == 20
    assert get_profile("mid_power_chip") is profile


def test_get_profile_rejects_unknown_name() -> None:
    """Looking up an unregistered profile raises KeyError listing the available names."""
    with pytest.raises(KeyError, match="Unknown hardware profile"):
        get_profile("definitely_not_a_real_target_xyz")


def test_list_profiles_filters_by_vendor_substring() -> None:
    """Vendor filtering matches a case-insensitive substring of the vendor name."""
    intel = list_profiles(vendor="intel")

    assert intel
    assert all("intel" in p.vendor.lower() for p in intel)


def test_load_toml_profiles_dir_returns_empty_for_missing_directory(tmp_path: Path) -> None:
    """Loading profiles from a non-existent directory yields an empty list, not an error."""
    missing = tmp_path / "no_such_profiles_dir"

    assert load_toml_profiles_dir(str(missing)) == []


def test_discover_platforms_registers_only_new_profiles() -> None:
    """Discovery hooks register fresh profiles once and skip already-known names."""
    fresh = HardwareProfile(
        name="discovered_chip",
        vendor="HookVendor",
        family="Auto",
        platform_class="custom",
        data_width=16,
        fraction=8,
    )
    duplicate = HardwareProfile(
        name="discovered_chip",
        vendor="HookVendor",
        family="Auto",
        platform_class="custom",
        data_width=16,
        fraction=8,
    )
    register_platform_hook(lambda: [fresh])
    register_platform_hook(lambda: [duplicate])

    discovered = discover_platforms()

    assert discovered.count("discovered_chip") == 1
    assert get_profile("discovered_chip") is fresh

    # Clean the global registry so re-runs stay deterministic.
    registry._PROFILES.pop("discovered_chip", None)
    registry._DISCOVERY_HOOKS.clear()
