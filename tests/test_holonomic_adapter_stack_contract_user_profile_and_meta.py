# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (user_profile_and_meta) from former test_holonomic_adapter_stack_contract.py

"""Tests of user profiles and package metadata in the adapter stack."""

from __future__ import annotations

from typing import Any, cast
import pytest
from sc_neurocore.adapters.base import (
    BaseStochasticAdapter,
)
from sc_neurocore.audio.user_profile import (
    Chronotype,
    UserProfile,
)


def test_base_adapter_is_abstract() -> None:
    """The base adapter cannot be instantiated."""
    with pytest.raises(TypeError):
        cast(Any, BaseStochasticAdapter)()


def test_user_profile_defaults() -> None:
    """A default profile is a bear chronotype at 10 Hz."""
    p = UserProfile()
    assert p.chronotype == Chronotype.BEAR
    assert p.get_best_target_hz() == 10.0
    assert "alpha" in p.sensitivity_map


def test_user_profile_explicit_target() -> None:
    """An explicit target frequency is used."""
    p = UserProfile(preferred_target_hz=7.5)
    assert p.get_best_target_hz() == 7.5


def test_user_profile_update_session() -> None:
    """Sessions are counted and update the preferred target."""
    p = UserProfile(chronotype=Chronotype.WOLF)
    p.update_from_session(avg_evs=60.0, peak_evs=80.0, best_target_hz=8.0)
    assert p.session_count == 1
    assert p.preferred_target_hz == 8.0
    p.update_from_session(avg_evs=55.0, peak_evs=70.0, best_target_hz=9.0)
    assert p.session_count == 2
    assert isinstance(p.preferred_target_hz, float)


def test_user_profile_update_low_evs() -> None:
    """A low score leaves no preferred target."""
    p = UserProfile()
    p.update_from_session(avg_evs=30.0, peak_evs=40.0, best_target_hz=5.0)
    assert p.preferred_target_hz is None


def test_user_profile_update_band_powers() -> None:
    """Session band powers update the baseline."""
    p = UserProfile()
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 10.0})
    assert "alpha" in p.baseline_band_powers
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 20.0})
    assert p.baseline_band_powers["alpha"] != 10.0


def test_user_profile_serialization() -> None:
    """A profile round-trips through serialisation."""
    p = UserProfile(user_id="test-1", chronotype=Chronotype.LION)
    d = p.to_dict()
    p2 = UserProfile.from_dict(d)
    assert p2.user_id == "test-1"
    assert p2.chronotype == Chronotype.LION


def test_user_profile_all_chronotypes() -> None:
    """Every chronotype has a positive target."""
    for chrono in Chronotype:
        p = UserProfile(chronotype=chrono)
        assert p.get_best_target_hz() > 0


def test_analysis_init_import() -> None:
    """The analysis package exports its concept mapper."""
    from sc_neurocore.analysis import SpikeToConceptMapper

    assert callable(SpikeToConceptMapper)


def test_chaos_init_import() -> None:
    """The chaos package exports its generator."""
    from sc_neurocore.chaos import ChaoticRNG

    assert callable(ChaoticRNG)
