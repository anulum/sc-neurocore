# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Rinzel co-simulation contracts

"""FitzHugh-Rinzel schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _fitzhugh_rinzel_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


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


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 FitzHugh-Rinzel co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((0.4, 7), (0.5, 8), (0.6, 8)),
        ids=("I=0.4", "I=0.5", "I=0.6"),
    )
    def test_fitzhugh_rinzel_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """FitzHugh-Rinzel has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state flow: four-stage
        simultaneous RK4 over the cubic fast membrane, linear recovery, and
        ultra-slow modulation equations; no reset; and rising-edge
        ``v >= v_threshold`` crossing detection. Over 3000 steps the hand model,
        schema runner, and emitted Q16.16 RTL produce 7, 8, and 8 crossings at
        ``I=0.4``, ``0.5``, and ``0.6`` respectively. This current band avoids the
        marginal ninth crossing at ``I=0.7``, where fixed-point rounding changes the
        spike count, so the contract states the robust band rather than hiding that
        boundary.
        """
        n_steps = 3000
        hand_spikes = _fitzhugh_rinzel_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("fitzhugh_rinzel", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("fitzhugh_rinzel", n_steps, current)
        assert hand_spikes == expected_spikes
        assert hand_spikes == py_spikes == vlog_spikes, (
            f"FitzHugh-Rinzel three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
