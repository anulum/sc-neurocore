# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Theta co-simulation contracts

"""Theta transcendental-LUT co-simulation contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _python_spike_count,
    _verilog_compiles,
    _verilog_spike_count,
)

_N_STEPS = 200
_INPUT_CURRENT = 50.0
_TRANSCENDENTAL_COSIM_MODELS = ["theta"]
_TRANSCENDENTAL_TOLERANCE_PCT = 5.0
_TRANSCENDENTAL_COMPILE_MODELS = ["theta"]


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestTranscendentalCoSimulation:
    """Auto model→RTL for transcendental models (exp/tanh/cosh via LUTs).

    Extends the polynomial cosim set: these models exercise the emitter's
    negative-LUT-literal handling, cosh support, and empty-parameter-list fix.
    """

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COSIM_MODELS)
    def test_both_produce_spikes(self, model_name: str) -> None:
        assert _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT) > 0
        assert _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT) > 0

    @pytest.mark.parametrize("model_name", _TRANSCENDENTAL_COSIM_MODELS)
    def test_spike_count_within_lut_tolerance(self, model_name: str) -> None:
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        assert py_spikes > 0 and vlog_spikes > 0
        gap_pct = abs(py_spikes - vlog_spikes) / max(py_spikes, 1) * 100
        assert gap_pct <= _TRANSCENDENTAL_TOLERANCE_PCT, (
            f"{model_name} transcendental co-sim gap {gap_pct:.1f}% exceeds "
            f"{_TRANSCENDENTAL_TOLERANCE_PCT}% (Python={py_spikes}, Verilog={vlog_spikes})"
        )

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
