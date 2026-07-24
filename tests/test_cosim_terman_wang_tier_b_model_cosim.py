# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTierBModelCosim from former test_cosim_terman_wang.py

"""Focused suite: TestTierBModelCosim from former test_cosim_terman_wang.py."""

from __future__ import annotations

from tests.cosim_terman_wang_support import *  # noqa: F403


class TestTierBModelCosim:
    """WC-A5 Tier-B Terman-Wang schema enrolment."""

    def test_terman_wang_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas track the hand oscillator over a varied drive.

        The 8,000-step sequence exercises the cubic fast nullcline, the ``tanh``
        recovery gate, external drive, all four simultaneous RK4 stages, and 28
        upward crossings followed by 28 re-arms. The hand model uses ``math.tanh``
        while the schema evaluator uses the NumPy transcendental, so state parity is
        asserted within a tight floating-point band rather than mislabelled as bit
        identity; spike decisions must still match exactly at every step.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = TermanWangOscillator()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "terman_wang.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "terman_wang.json")
        currents = (-1.0, 0.0, 0.5, 0.25, -0.5, 0.75, 0.0, 0.4) * 1000
        spike_count = 0
        rearm_count = 0
        was_above = hand.v >= hand.v_peak

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            now_above = hand.v >= hand.v_peak
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == pytest.approx(expected, rel=1e-12, abs=1e-10)
                assert json_schema.state[variable] == pytest.approx(expected, rel=1e-12, abs=1e-10)

        assert spike_count == 28
        assert rearm_count == 28
