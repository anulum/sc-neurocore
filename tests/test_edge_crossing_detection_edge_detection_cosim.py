# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEdgeDetectionCosim from former test_edge_crossing_detection.py

"""Focused suite: TestEdgeDetectionCosim from former test_edge_crossing_detection.py."""

from __future__ import annotations

from tests.edge_crossing_detection_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestEdgeDetectionCosim:
    """End-to-end Python-runner vs emitted-RTL parity for edge-crossing oscillators."""

    def test_fitzhugh_nagumo_faithful_three_way_q1616_parity(self) -> None:
        """Faithful FitzHugh-Nagumo co-simulates at exact Q16.16 spike-count parity.

        The RK4, no-reset, crossing-detection schema reproduces ``FitzHughNagumoNeuron``
        bit-for-bit in float64, and the emitted Q16.16 RTL — using the new ``_thr_prev``
        rising-edge datapath — reproduces the same sustained relaxation-oscillation spike
        train exactly (8 of 3000 steps at ``I=0.5``), a partial train that exercises
        repeated crossing re-arming rather than a single event. The cubic right-hand side
        is polynomial, so no look-up table is involved and the parity is bit-exact.
        """
        current, n_steps = 0.5, 3000
        hand = _fhn_hand()
        hand_spikes = sum(hand.step(current) for _ in range(n_steps))

        schema = UniversalNeuron.from_dict(_fhn_schema("crossing"))
        schema_spikes = sum(1 for _ in range(n_steps) if schema.step(I=current))

        verilog_spikes = _verilog_spike_count_q1616(
            UniversalNeuron.from_dict(_fhn_schema("crossing")), n_steps, current, "fhn"
        )

        assert 1 < schema_spikes < n_steps  # a repetitive partial train
        assert hand_spikes == schema_spikes == verilog_spikes

    def test_faithful_fitzhugh_nagumo_schema_matches_hand_state_sequence(self) -> None:
        """The faithful schema mirrors the hand model's spike decision and both states."""
        hand = _fhn_hand()
        schema = UniversalNeuron.from_dict(_fhn_schema("crossing"))
        for current in (0.5, 0.5, 0.0, 0.6, 0.4):
            for _ in range(200):
                assert int(bool(schema.step(I=current))) == hand.step(current)
                assert schema.state["v"] == hand.v
                assert schema.state["w"] == hand.w
