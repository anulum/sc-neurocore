# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTierBModelCosim from former test_cosim_fitzhugh_rinzel.py

"""Focused suite: TestTierBModelCosim from former test_cosim_fitzhugh_rinzel.py."""

from __future__ import annotations

from tests.cosim_fitzhugh_rinzel_support import *  # noqa: F403


class TestTierBModelCosim:
    """WC-A5 Tier-B FitzHugh-Rinzel schema enrolment."""

    def test_fitzhugh_rinzel_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas reproduce the hand model over a varied drive.

        The 1,200-step sequence alternates quiet, depolarising, and negative currents,
        exercising every RK4 stage in the three coupled equations, one upward crossing,
        and subsequent below-threshold re-arming. Exact state equality is required for
        ``v``, ``w``, and the ultra-slow ``y`` variable after every step, so either
        schema format drifting in an initial value, parameter, equation, operation order,
        or no-reset crossing decision fails immediately.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = FitzHughRinzelNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "fitzhugh_rinzel.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "fitzhugh_rinzel.json")
        currents = (0.0, 0.17, 0.5, 0.31, 0.83, -0.07) * 200
        spike_count = 0
        rearmed = False

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            if spike_count and hand.v < hand.v_threshold:
                rearmed = True
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w", "y"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 1
        assert rearmed
