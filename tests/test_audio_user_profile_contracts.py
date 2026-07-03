# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Audio user-profile contract tests

"""Focused contracts for adaptive-audio user-profile persistence."""

from __future__ import annotations

from pytest import approx

from sc_neurocore.audio.user_profile import Chronotype, UserProfile


def test_user_profile_explicit_target_overrides_chronotype_default() -> None:
    """An explicit preferred target frequency takes precedence over chronotype."""
    profile = UserProfile(chronotype=Chronotype.WOLF, preferred_target_hz=7.5)

    assert profile.get_best_target_hz() == 7.5


def test_user_profile_adopts_and_smooths_high_evs_targets() -> None:
    """High-EVS sessions adopt the first target and smooth later targets."""
    profile = UserProfile()

    profile.update_from_session(avg_evs=65.0, peak_evs=82.0, best_target_hz=6.0)
    profile.update_from_session(avg_evs=72.0, peak_evs=90.0, best_target_hz=9.0)

    assert profile.session_count == 2
    assert profile.preferred_target_hz == approx(6.9)


def test_user_profile_band_powers_initialise_and_blend_by_band() -> None:
    """Band-power updates initialise on first write and EMA-blend thereafter."""
    profile = UserProfile()

    profile.update_from_session(
        avg_evs=20.0,
        peak_evs=30.0,
        band_powers={"alpha": 2.0, "beta": 4.0},
    )
    profile.update_from_session(
        avg_evs=25.0,
        peak_evs=35.0,
        band_powers={"alpha": 12.0, "gamma": 5.0},
    )

    assert profile.baseline_band_powers["alpha"] == approx(4.0)
    assert profile.baseline_band_powers["beta"] == approx(4.0)
    assert profile.baseline_band_powers["gamma"] == approx(5.0)


def test_user_profile_roundtrip_preserves_custom_state() -> None:
    """Dictionary round-trips preserve custom profile maps and counters."""
    profile = UserProfile(
        user_id="study-7",
        chronotype=Chronotype.LION,
        baseline_band_powers={"alpha": 3.0},
        preferred_cost_weights={"w_micro": 1.1, "w_reg": 0.02},
        sensitivity_map={"alpha": 1.2},
        session_count=4,
        preferred_target_hz=11.0,
    )

    restored = UserProfile.from_dict(profile.to_dict())

    assert restored.user_id == "study-7"
    assert restored.chronotype is Chronotype.LION
    assert restored.baseline_band_powers == {"alpha": 3.0}
    assert restored.preferred_cost_weights == {"w_micro": 1.1, "w_reg": 0.02}
    assert restored.sensitivity_map == {"alpha": 1.2}
    assert restored.session_count == 4
    assert restored.preferred_target_hz == 11.0
