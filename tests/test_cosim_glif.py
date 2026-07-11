# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GLIF co-simulation contracts

"""GLIF schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _glif_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


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


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 GLIF co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (15.0, 0), (22.0, 23), (30.0, 54), (45.0, 86), (50.0, 95)),
        ids=("rest", "subthreshold", "onset", "tonic", "high-drive", "strong-drive"),
    )
    def test_glif_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """GLIF has exact hand/schema/Q16.16 spike-count parity across six regimes.

        The schema mirrors the maintained four-state, candidate-first classical-RK4
        hand model with level ``v >= theta`` detection and adaptive reset. Hand model
        and schema runner agree exactly at every operating point. The compiler lowers
        reset expressions from the integrated candidate and exposes the same post-reset
        state in RTL, so Q16.16 preserves the complete spike count despite quantising
        ``a_theta=0.01`` and the adaptive increments. Rest, subthreshold, onset, tonic,
        and high-drive regimes are all enrolled rather than one selected current.
        """
        n_steps = 1000
        hand_spikes = _glif_hand_spike_count(n_steps, current)
        schema_spikes = _python_spike_count("glif", n_steps, current)
        verilog_spikes = _verilog_spike_count_q1616("glif", n_steps, current)

        assert hand_spikes == schema_spikes == verilog_spikes == expected_spikes, (
            f"GLIF exact Q16.16 mismatch at I={current}: "
            f"hand={hand_spikes}, schema={schema_spikes}, verilog={verilog_spikes}"
        )
