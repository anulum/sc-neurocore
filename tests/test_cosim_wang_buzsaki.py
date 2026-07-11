# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Wang-Buzsáki co-simulation contracts

"""Wang-Buzsáki schema, hand-model, and Q16.16 RTL parity contracts."""

from __future__ import annotations

import pytest

from tests.cosim_support import (
    HAS_IVERILOG,
    _wang_buzsaki_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count_q1616,
)


@pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
class TestQ1616Precision:
    """Q16.16 Wang-Buzsáki co-simulation fidelity."""

    def test_wang_buzsaki_q1616_macrostep_parity(self) -> None:
        """Faithful macro-step Wang-Buzsaki: hand == schema exact, verilog within one spike.

        The re-enrolled schema mirrors ``WangBuzsakiNeuron``'s maintained integrator: a
        sequential (Gauss-Seidel) forward Euler with ``substeps=50`` (50 inner ``dt=0.01``
        sub-steps per 0.5 ms macro step, the gating variables ``h``/``n`` updated from the old
        voltage and the membrane voltage ``v`` from the new gates) and a rising-edge
        ``v >= v_threshold`` crossing evaluated only on the macro boundary, no reset. The
        earlier schema was single-step ``method="euler"`` with a sigmoid-caricature ``m_inf``
        and unfaithful gate initial conditions, so it could only be compared schema-vs-verilog
        under a 15% band; the macro-step schema now reproduces the hand model's
        action-potential count exactly, so **hand == schema** (one hand ``step()`` per schema
        macro ``step()``). Unlike Hodgkin-Huxley (simultaneous RK4), Wang-Buzsaki requires the
        DSL's ``gauss_seidel`` mode — the hand model updates the gates before the voltage, and
        simultaneous Euler drifts.

        The Q16.16 RTL runs 50 clocks per macro step (one sequential sub-step each, the crossing
        gated to the macro boundary) and tracks the schema **within one spike** over the bounded
        window. Wang-Buzsaki's exprel gating and its ``m_inf = alpha_m/(alpha_m+beta_m)``
        runtime division lower to a 256-entry look-up table plus a fixed-point divide; the
        fixed-point trajectory drifts from float64 and the drift is look-up-table- and
        fixed-point-resolution-limited, not a tolerance knob — three-way exact over this
        bounded window and accumulating beyond it, an honest per-model hardware-fidelity band.
        """
        current, macro_steps, substeps = 10.0, 20, 50
        hand_spikes = _wang_buzsaki_hand_spike_count(macro_steps, current)
        py_spikes = _python_spike_count("wang_buzsaki", macro_steps, current)
        vlog_spikes = _verilog_spike_count_q1616("wang_buzsaki", macro_steps * substeps, current)
        assert 1 < py_spikes < macro_steps  # a partial macro-step train, not saturated
        assert hand_spikes == py_spikes, (
            f"Wang-Buzsaki hand/schema macro-step mismatch: hand={hand_spikes}, schema={py_spikes}"
        )
        assert abs(py_spikes - vlog_spikes) <= 1, (
            f"Wang-Buzsaki Q16.16 macro-step gap > 1 spike "
            f"(schema={py_spikes}, verilog={vlog_spikes})"
        )
