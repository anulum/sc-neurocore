# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Pernarowski co-simulation contracts

"""Pernarowski schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _pernarowski_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


class TestTierBModelCosim:
    """WC-A5 Tier-B Pernarowski schema enrolment."""

    def test_pernarowski_schema_formats_match_hand_rk4_sequence(self) -> None:
        """The TOML and JSON schemas reproduce the hand model over a varied drive.

        The 5,000-step sequence exercises the external-current term and every RK4
        stage across the fast cubic coordinate, recovery variable, and ultra-slow
        adaptation variable. It also covers 17 upward crossings and 17 subsequent
        below-threshold re-arms. Exact state equality is required after every step,
        so either schema format drifting in initial state, parameters, equations,
        operation order, or no-reset edge detection fails immediately.
        """
        schema_dir = Path(__file__).resolve().parents[1] / "src/sc_neurocore/neurons/model_schemas"
        hand = PernarowskiNeuron()
        toml_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.toml")
        json_schema = UniversalNeuron.from_schema(schema_dir / "pernarowski.json")
        currents = (0.0, 0.1, -0.1, 0.2, 0.0, -0.2, 0.15, 0.05) * 625
        spike_count = 0
        rearm_count = 0
        was_above = hand.v >= hand.v_threshold

        for current in currents:
            hand_spike = hand.step(current)
            spike_count += hand_spike
            now_above = hand.v >= hand.v_threshold
            if was_above and not now_above:
                rearm_count += 1
            was_above = now_above
            assert int(bool(toml_schema.step(I=current))) == hand_spike
            assert int(bool(json_schema.step(I=current))) == hand_spike
            for variable in ("v", "w", "z"):
                expected = getattr(hand, variable)
                assert toml_schema.state[variable] == expected
                assert json_schema.state[variable] == expected

        assert spike_count == 17
        assert rearm_count == 17


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Pernarowski co-simulation fidelity."""

    @pytest.mark.parametrize(
        "current",
        (-0.1, 0.0, 0.1, 0.2),
        ids=("I=-0.1", "I=0.0", "I=0.1", "I=0.2"),
    )
    def test_pernarowski_q1616_parity(self, current: float) -> None:
        """Pernarowski has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained three-state beta-cell flow:
        simultaneous four-stage RK4 over the cubic fast coordinate and two
        separated slow variables, rising-edge ``v >= v_threshold`` detection,
        and no reset. The oscillator is autonomous, so input current shifts the
        trajectory rather than gating a silent/single/train transition. At each
        enrolled point from ``I=-0.1`` through ``I=0.2``, the hand model, schema
        runner, and emitted Q16.16 RTL report 17 crossings over 5,000 steps.
        """
        n_steps = 5000
        hand_spikes = _pernarowski_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("pernarowski", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("pernarowski", n_steps, current)
        assert 1 < hand_spikes < n_steps
        assert hand_spikes == py_spikes == vlog_spikes == 17, (
            f"Pernarowski three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )
