# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCoSimulation from former test_cosim_emitters.py

"""Focused suite: TestCoSimulation from former test_cosim_emitters.py."""

from __future__ import annotations

from tests.cosim_emitters_support import *  # noqa: F403


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestCoSimulation:
    """Python ↔ Verilog co-simulation: validate spike behaviour equivalence."""

    @pytest.mark.parametrize("model_name", _COSIM_MODELS)
    def test_both_produce_spikes(self, model_name: str) -> None:
        """Verify both implementations produce non-zero spike output."""
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)

        # Both should spike (model is being driven with sufficient current)
        assert py_spikes > 0, f"Python {model_name} produced 0 spikes"
        assert vlog_spikes > 0, f"Verilog {model_name} produced 0 spikes"

    @pytest.mark.parametrize("model_name", _COSIM_MODELS)
    def test_spike_count_accuracy(self, model_name: str) -> None:
        """Q8.8 is exact except for the declared Izhikevich one-spike boundary.

        Candidate-first reset semantics expose a single marginal Izhikevich
        crossing at this coarse precision: float64 reports 25 spikes and Q8.8
        reports 24. The same model is exact at Q16.16 below. Every other baseline
        model retains exact Q8.8 parity.
        """
        py_spikes = _python_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)
        vlog_spikes = _verilog_spike_count(model_name, _N_STEPS, _INPUT_CURRENT)

        assert py_spikes > 0, f"Python {model_name} must spike"
        assert vlog_spikes > 0, f"Verilog {model_name} must spike"

        gap = abs(py_spikes - vlog_spikes)
        gap_pct = gap / max(py_spikes, 1) * 100
        print(
            f"\n  Co-sim {model_name}: Python={py_spikes}, Verilog={vlog_spikes}, "
            f"gap={gap} ({gap_pct:.1f}%)"
        )

        if model_name == "izhikevich":
            assert (py_spikes, vlog_spikes) == (25, 24)
        else:
            assert gap == 0, (
                f"Q8.8 co-simulation must be exact: {gap_pct:.1f}% "
                f"(model={model_name}, Python={py_spikes}, Verilog={vlog_spikes})"
            )

    @pytest.mark.parametrize("model_name", [m for m in _COSIM_MODELS if m != "izhikevich"])
    def test_no_current_no_spikes(self, model_name: str) -> None:
        """With zero input current, linear models should not spike.

        Izhikevich is excluded: its +140 constant term drives intrinsic dynamics,
        and Q8.8 quantization of 0.04*v^2 (-65^2=4225, overflows 16-bit product)
        causes divergent behaviour at zero current. Use Q16.16 for Izhikevich
        if zero-current silence is required.
        """
        py_spikes = _python_spike_count(model_name, 50, 0.0)
        vlog_spikes = _verilog_spike_count(model_name, 50, 0.0)

        assert py_spikes == 0, f"Python {model_name} spiked with zero current"
        assert vlog_spikes == 0, f"Verilog {model_name} spiked with zero current"

    def test_python_sim_is_deterministic(self) -> None:
        """Verify Python simulation is deterministic across runs."""
        a = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        b = _python_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        assert a == b

    def test_verilog_sim_is_deterministic(self) -> None:
        """Verify Verilog simulation is deterministic across runs."""
        a = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        b = _verilog_spike_count("lif", _N_STEPS, _INPUT_CURRENT)
        assert a == b
