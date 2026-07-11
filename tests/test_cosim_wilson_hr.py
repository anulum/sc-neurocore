# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wilson-HR co-simulation contracts

"""Wilson-HR schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _wilson_hr_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


class TestTierBModelCosim:
    """WC-A5 Tier-B Wilson-HR schema enrolment."""

    def test_wilson_hr_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas track Wilson-HR over a varied drive.

        Five passes through eight 100-step drive blocks exercise the polynomial
        membrane nullcline, coupled recovery flow, all four simultaneous RK4 stages,
        and 35 hard voltage resets. Both schema formats must reproduce every hand-model
        spike decision and both post-step states exactly; equality of ``r`` on spiking
        steps guards the contract that only ``v`` resets.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = WilsonHRNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "wilson_hr.json")
        current_blocks = (0.0, 10.0, 2.0, 10.0, 0.0, 5.0, 10.0, 2.0)
        spike_count = 0
        reset_count = 0

        for _cycle in range(5):
            for current in current_blocks:
                for _step in range(100):
                    hand_spike = hand.step(current)
                    spike_count += hand_spike
                    if hand_spike:
                        assert hand.v == -0.7
                        reset_count += 1
                    assert int(bool(toml_schema.step(I=current))) == hand_spike
                    assert int(bool(json_schema.step(I=current))) == hand_spike
                    for variable in ("v", "r"):
                        expected = getattr(hand, variable)
                        assert toml_schema.state[variable] == expected
                        assert json_schema.state[variable] == expected

        assert spike_count == 35
        assert reset_count == 35


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Wilson-HR co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.0, 0), (2.0, 1), (10.0, 4)),
        ids=("silent", "single-spike", "four-spike-train"),
    )
    def test_wilson_hr_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Wilson-HR has exact three-way Q16.16 spike-count parity.

        The schema mirrors the maintained two-state polynomial cortical model:
        simultaneous four-stage RK4 over ``v`` and ``r``, level detection at
        ``v >= v_peak``, and a hard ``v = -0.7`` reset that preserves the candidate
        recovery state. Over 5,000 steps the hand model, schema runner, and emitted
        RTL reproduce the silent, single-spike, and four-spike operating points.
        """
        n_steps = 5000
        hand_spikes = _wilson_hr_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("wilson_hr", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wilson_hr", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Wilson-HR three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
