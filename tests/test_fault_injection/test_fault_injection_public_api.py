# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for sc_neurocore.fault_injection package public API

"""Tests for the `sc_neurocore.fault_injection` package re-exports.

Antigravity authored `fault_injection.py` but did not wire
`__init__.py` to re-export the 6 public symbols. Arcane Sapience
wired the re-exports in this batch (audit B1 pkg 2 of #57).
This test pins the contract.
"""

from __future__ import annotations

import sc_neurocore.fault_injection as fi
from sc_neurocore.fault_injection import fault_injection as inner


SYMBOLS: tuple[str, ...] = (
    "FaultInjectionResult",
    "FaultInjector",
    "FaultModel",
    "RadiationProfile",
    "ResilienceBenchmark",
    "ResilienceReport",
)


def test_tier_is_industrial() -> None:
    assert fi.__tier__ == "industrial"


def test_all_lists_six_symbols() -> None:
    assert isinstance(fi.__all__, list)
    assert len(fi.__all__) == 6
    assert set(fi.__all__) == set(SYMBOLS)


def test_symbols_importable_from_package() -> None:
    for sym in SYMBOLS:
        assert hasattr(fi, sym), f"package missing {sym!r}"


def test_symbols_identity_with_inner() -> None:
    """Top-level package symbol IS the inner-module symbol."""
    for sym in SYMBOLS:
        assert getattr(fi, sym) is getattr(inner, sym)


def test_fault_model_enum_has_five_members() -> None:
    """FaultModel enum: bit-flip, stuck-at-0, stuck-at-1, gaussian noise, dropout."""
    members = {m.name for m in fi.FaultModel}
    assert {"BIT_FLIP", "STUCK_AT_0", "STUCK_AT_1", "GAUSSIAN_NOISE", "DROPOUT"} == members


def test_radiation_profile_presets() -> None:
    """The four published radiation environments are constructible.

    BER ordering must respect physics: terrestrial < LEO < GEO < deep space.
    """
    leo = fi.RadiationProfile.leo()
    geo = fi.RadiationProfile.geo()
    deep = fi.RadiationProfile.deep_space()
    terrestrial = fi.RadiationProfile.terrestrial()

    assert leo.name == "LEO"
    assert geo.name == "GEO"
    assert deep.name == "Deep Space"
    assert terrestrial.name == "Terrestrial"

    # Physics ordering: thermal neutron background < LEO < GEO < interplanetary cosmic rays
    assert terrestrial.ber < leo.ber < geo.ber < deep.ber


def test_radiation_profile_published_bers() -> None:
    """The published BER constants are within the documented orders of magnitude.

    From `fault_injection.py` lines 43–58:
        terrestrial 1e-10, LEO 1e-7, GEO 5e-6, deep space 1e-4
    """
    assert fi.RadiationProfile.terrestrial().ber == 1e-10
    assert fi.RadiationProfile.leo().ber == 1e-7
    assert fi.RadiationProfile.geo().ber == 5e-6
    assert fi.RadiationProfile.deep_space().ber == 1e-4
