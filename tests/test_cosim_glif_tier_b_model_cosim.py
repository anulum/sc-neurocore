# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTierBModelCosim from former test_cosim_glif.py

"""Focused suite: TestTierBModelCosim from former test_cosim_glif.py."""

from __future__ import annotations

from tests.cosim_glif_support import *  # noqa: F403


class TestTierBModelCosim:
    """WC-A5 Tier-B GLIF schema enrolment."""

    def test_glif_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The paired GLIF schemas reproduce every hand-model RK4 state and reset.

        The 4,000-step varied drive exercises all four coupled linear equations,
        every RK4 stage, silence, tonic firing, and 181 candidate-first adaptive
        resets. Exact state and event equality after every step catches drift in
        either schema format's integration method, threshold relation, parameter,
        reset source, or post-candidate update order.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = GLIFNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "glif.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "glif.json")
        currents = (0.0, 15.0, 22.0, 30.0, 45.0, 50.0, 30.0, 22.0) * 500
        spike_count = 0
        reset_count = 0

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            if hand_spike:
                assert hand.v == hand.v_reset
                reset_count += 1
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "theta", "i_asc1", "i_asc2"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == reset_count == 181
