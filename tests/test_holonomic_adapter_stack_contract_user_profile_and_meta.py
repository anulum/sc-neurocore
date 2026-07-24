# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (user_profile_and_meta) from former test_holonomic_adapter_stack_contract.py

from __future__ import annotations

from tests.holonomic_adapter_stack_contract_support import *  # noqa: F403

def test_base_adapter_is_abstract():
    with pytest.raises(TypeError):
        BaseStochasticAdapter()


def test_user_profile_defaults():
    p = UserProfile()
    assert p.chronotype == Chronotype.BEAR
    assert p.get_best_target_hz() == 10.0
    assert "alpha" in p.sensitivity_map


def test_user_profile_explicit_target():
    p = UserProfile(preferred_target_hz=7.5)
    assert p.get_best_target_hz() == 7.5


def test_user_profile_update_session():
    p = UserProfile(chronotype=Chronotype.WOLF)
    p.update_from_session(avg_evs=60.0, peak_evs=80.0, best_target_hz=8.0)
    assert p.session_count == 1
    assert p.preferred_target_hz == 8.0
    p.update_from_session(avg_evs=55.0, peak_evs=70.0, best_target_hz=9.0)
    assert p.session_count == 2
    assert isinstance(p.preferred_target_hz, float)


def test_user_profile_update_low_evs():
    p = UserProfile()
    p.update_from_session(avg_evs=30.0, peak_evs=40.0, best_target_hz=5.0)
    assert p.preferred_target_hz is None


def test_user_profile_update_band_powers():
    p = UserProfile()
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 10.0})
    assert "alpha" in p.baseline_band_powers
    p.update_from_session(avg_evs=60.0, peak_evs=70.0, band_powers={"alpha": 20.0})
    assert p.baseline_band_powers["alpha"] != 10.0


def test_user_profile_serialization():
    p = UserProfile(user_id="test-1", chronotype=Chronotype.LION)
    d = p.to_dict()
    p2 = UserProfile.from_dict(d)
    assert p2.user_id == "test-1"
    assert p2.chronotype == Chronotype.LION


def test_user_profile_all_chronotypes():
    for chrono in Chronotype:
        p = UserProfile(chronotype=chrono)
        assert p.get_best_target_hz() > 0


def test_analysis_init_import():
    from sc_neurocore.analysis import SpikeToConceptMapper

    assert callable(SpikeToConceptMapper)


def test_chaos_init_import():
    from sc_neurocore.chaos import ChaoticRNG

    assert callable(ChaoticRNG)


