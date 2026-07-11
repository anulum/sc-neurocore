# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Terman-Wang co-simulation contracts

"""Terman-Wang schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _terman_wang_hand_spike_count,
    _python_spike_count,
    _verilog_compiles,
    _verilog_spike_count_q1616,
)

_TRANSCENDENTAL_COMPILE_MODELS = ["terman_wang"]


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


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Terman-Wang co-simulation fidelity."""

    @pytest.mark.parametrize(
        ("current", "expected_spikes"),
        ((-1.0, 0), (0.0, 1), (0.5, 3)),
        ids=("silent", "single-crossing", "oscillatory-train"),
    )
    def test_terman_wang_q1616_parity(self, current: float, expected_spikes: int) -> None:
        """Terman-Wang has exact three-way Q16.16 spike-count parity.

        The enrolled schema mirrors the maintained two-state LEGION oscillator:
        simultaneous four-stage RK4 over the cubic fast nullcline and ``tanh``-gated
        slow recovery, rising-edge ``v >= v_peak`` detection, and no reset. The
        transcendental gate makes raw state bit identity non-portable, so the declared
        observable is the robust silent/single/train crossing count: 0, 1, and 3 at
        ``I=-1.0``, ``0.0``, and ``0.5`` respectively over 8,000 steps.
        """
        n_steps = 8000
        hand_spikes = _terman_wang_hand_spike_count(n_steps, current)
        py_spikes = _python_spike_count("terman_wang", n_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("terman_wang", n_steps, current)
        assert hand_spikes == py_spikes == vlog_spikes == expected_spikes, (
            f"Terman-Wang three-way mismatch at I={current}: hand={hand_spikes}, "
            f"schema={py_spikes}, verilog={vlog_spikes}"
        )


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestTranscendentalCoSimulation:
    """Auto model→RTL compile contracts for Terman-Wang."""

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COMPILE_MODELS)
    def test_transcendental_model_lowers_to_valid_verilog(self, model_name: str) -> None:
        """Non-baseline models lower to iverilog-valid Verilog without malformed literals.

        This is the emitter-fix verification: before the negative-LUT-literal,
        cosh, and empty-parameter fixes these models either raised
        "Unsupported function" or emitted malformed `W'sd-N` literals. The GLIF
        Q8.8 path is resolution-limited and the conductance-model look-up tables can
        be too coarse for the dedicated Q16.16 behavioural claims, so this assertion
        covers valid synthesisable RTL rather than spike parity.
        """
        verilog = UniversalNeuron.from_schema(model_name).to_verilog(module_name=f"sc_{model_name}")
        assert "'sd-" not in verilog  # no malformed negative literals
        assert _verilog_compiles(model_name)
