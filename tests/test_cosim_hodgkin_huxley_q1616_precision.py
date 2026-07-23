# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestQ1616Precision from former test_cosim_hodgkin_huxley.py

"""Focused suite: TestQ1616Precision from former test_cosim_hodgkin_huxley.py."""

from __future__ import annotations

from tests.cosim_hodgkin_huxley_support import *  # noqa: F403

@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Hodgkin-Huxley co-simulation fidelity."""

    def test_hodgkin_huxley_q1616_macrostep_parity(self) -> None:
        """Faithful macro-step Hodgkin-Huxley: hand == schema exact, verilog within one spike.

        The re-enrolled schema mirrors ``HodgkinHuxleyNeuron(integrator="rk4")``'s maintained
        integrator: RK4 with ``substeps=100`` (100 inner ``dt=0.01`` sub-steps per 1 ms macro
        step) and a rising-edge (``v >= v_threshold``) crossing evaluated only on the macro
        boundary, no reset. The earlier schema was single-step ``method="euler"`` — neither the
        hand model's RK4 nor its macro-stepping — so it could only be compared schema-vs-verilog
        under a 5% band; the macro-step schema now reproduces the hand model's action-potential
        count exactly, so **hand == schema** (one hand ``step()`` per schema macro ``step()``).
        The comparison is against the ``integrator="rk4"`` (simultaneous) path, not the
        Gauss-Seidel default ``baseline_euler``, which the DSL's simultaneous integration matches.

        The Q16.16 RTL runs 100 clocks per macro step (one integration sub-step each, the
        crossing gated to the macro boundary) and tracks the schema **within one spike** over the
        bounded window. Like the stiff six-state Connor-Stevens (and unlike the well-conditioned
        Morris-Lecar), Hodgkin-Huxley's exprel / sigmoid gating lowers to 256-entry look-up
        tables; the fixed-point trajectory drifts from float64 and the drift is
        **look-up-table-resolution-limited, not datapath-precision-limited**, so it holds
        three-way over a bounded window and accumulates beyond it — an honest per-model
        hardware-fidelity band, not a tolerance knob. The macro-step lowering itself is bit-exact
        (proven on the polynomial FitzHugh-Nagumo sub-step cosim); the residual is genuine
        conductance-LUT quantisation.
        """
        current, macro_steps, substeps = 15.0, 20, 100
        hand_spikes = _hodgkin_huxley_hand_spike_count(macro_steps, current)
        py_spikes = _python_spike_count("hodgkin_huxley", macro_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("hodgkin_huxley", macro_steps * substeps, current)
        assert 1 < py_spikes < macro_steps  # a partial macro-step train, not saturated
        assert hand_spikes == py_spikes, (
            f"Hodgkin-Huxley hand/schema macro-step mismatch: hand={hand_spikes}, schema={py_spikes}"
        )
        assert abs(py_spikes - vlog_spikes) <= 1, (
            f"Hodgkin-Huxley Q16.16 macro-step gap > 1 spike "
            f"(schema={py_spikes}, verilog={vlog_spikes})"
        )
