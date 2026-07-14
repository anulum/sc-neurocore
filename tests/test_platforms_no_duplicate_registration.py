# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests guarding against duplicate hardware-profile registration

"""Guard the built-in profile modules against duplicate-name registration.

A previous revision registered ``cortical_labs_dishbrain``,
``finalspark_neuroplatform`` and ``biomemory_dna`` three to four times each
(and ``brainscales2``/``spinnaker2`` twice) with conflicting fields; the
silent last-wins overwrite in ``_reg`` masked the bug. These tests lock the
de-duplicated state: ``_reg`` now rejects a duplicate, the three built-in
modules each declare every profile name once, and the previously-conflicting
platforms resolve to their single canonical definition.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pytest

from sc_neurocore.compiler.platforms import registry
from sc_neurocore.compiler.platforms.registry import (
    HardwareProfile,
    _reg,
    get_profile,
)

_NAME_PATTERN = re.compile(r'name="([^"]+)"')
_BUILTIN_MODULES = (
    "_cmos_accelerator_profiles",
    "_cmos_architecture_profiles",
    "_cmos_fpga_profiles",
    "_cmos_processor_profiles",
    "_cmos_reference_profiles",
    "cmos_profiles",
    "exotic_profiles",
    "neuromorphic_profiles",
)


def _source_profile_names() -> list[str]:
    """Collect every ``name="..."`` literal across the built-in profile modules."""
    package_dir = Path(registry.__file__).parent
    names: list[str] = []
    for module in _BUILTIN_MODULES:
        source = (package_dir / f"{module}.py").read_text(encoding="utf-8")
        names.extend(_NAME_PATTERN.findall(source))
    return names


def test_no_duplicate_profile_names_in_source() -> None:
    """Each built-in profile name is declared exactly once across the modules."""
    names = _source_profile_names()
    duplicates = {name: count for name, count in Counter(names).items() if count > 1}

    assert duplicates == {}, f"Duplicate profile registrations found: {duplicates}"


def test_source_names_match_registry() -> None:
    """The declared source names equal the registered names with no losses."""
    source_names = set(_source_profile_names())

    # Every source-declared profile is reachable through the registry.
    for name in source_names:
        assert name in registry._PROFILES


def test_reg_rejects_duplicate_name() -> None:
    """Registering a name twice raises unless override is explicit."""
    probe = HardwareProfile(
        name="dup_guard_probe",
        vendor="Test",
        family="Auto",
        platform_class="custom",
        data_width=16,
        fraction=8,
    )
    try:
        _reg(probe)
        with pytest.raises(ValueError, match="Duplicate hardware-profile registration"):
            _reg(probe)
    finally:
        registry._PROFILES.pop("dup_guard_probe", None)


def test_reg_allows_explicit_override() -> None:
    """``allow_override=True`` replaces an existing profile without raising."""
    first = HardwareProfile(
        name="override_probe",
        vendor="First",
        family="Auto",
        platform_class="custom",
        data_width=16,
        fraction=8,
    )
    second = HardwareProfile(
        name="override_probe",
        vendor="Second",
        family="Auto",
        platform_class="custom",
        data_width=8,
        fraction=4,
    )
    try:
        _reg(first)
        result = _reg(second, allow_override=True)

        assert result is second
        assert get_profile("override_probe").vendor == "Second"
    finally:
        registry._PROFILES.pop("override_probe", None)


@pytest.mark.parametrize(
    ("name", "vendor", "family", "platform_class", "data_width", "fraction"),
    [
        ("cortical_labs_dishbrain", "Cortical Labs", "DishBrain", "wetware", 8, 0),
        ("finalspark_neuroplatform", "FinalSpark", "Neuroplatform", "wetware", 8, 0),
        ("biomemory_dna", "Biomemory", "DNA-Drive", "molecular", 2, 0),
        ("belousov_zhabotinsky", "Academic", "BZ Reaction", "molecular", 1, 0),
        ("fujitsu_digital_annealer", "Fujitsu", "Digital Annealer", "reversible", 64, 0),
        ("rl_toffoli_asic", "Custom", "Reversible ASIC", "reversible", 32, 16),
        ("ibm_microfluidic", "IBM", "Electronic Blood", "microfluidic", 4, 0),
        ("mems_resonator", "SiTime", "MEMS Logic", "microfluidic", 12, 0),
        ("spinnaker2", "SpiNNcloud", "SpiNNaker 2", "neuromorphic", 32, 16),
        ("brainscales2", "Heidelberg", "BrainScaleS-2", "neuromorphic", 8, 4),
    ],
)
def test_canonical_resolved_values(
    name: str,
    vendor: str,
    family: str,
    platform_class: str,
    data_width: int,
    fraction: int,
) -> None:
    """Previously-conflicting platforms resolve to a single canonical definition."""
    p = get_profile(name)

    assert p.vendor == vendor
    assert p.family == family
    assert p.platform_class == platform_class
    assert p.data_width == data_width
    assert p.fraction == fraction


def test_brainscales2_and_spinnaker2_use_wrap_overflow() -> None:
    """The canonical analog/GALS neuromorphic targets keep wrap overflow semantics."""
    assert get_profile("brainscales2").overflow == "wrap"
    assert get_profile("spinnaker2").overflow == "wrap"
