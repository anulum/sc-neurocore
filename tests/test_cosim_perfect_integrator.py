# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator co-simulation contracts

"""Perfect Integrator schema, hand-model, and RTL parity contracts."""

from __future__ import annotations

from pathlib import Path

import pytest

from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _perfect_integrator_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count,
    simulate,
)

_N_STEPS = 1_000
_INPUT_CURRENT = 0.7
ROOT = Path(__file__).resolve().parents[1]


class TestTierBModelCosim:
    """WC-A5 Tier-B Perfect Integrator enrolment."""

    def test_perfect_integrator_schema_matches_hand_model_sequence(self) -> None:
        """The schema mirrors the hand-authored non-leaky integrator step law."""
        hand = PerfectIntegratorNeuron.naud_gerstner_2012()
        schema = UniversalNeuron.from_schema("perfect_integrator")

        for current in (0.0, 2.0, 5.0, 3.0, 10.0, 1.0):
            assert schema.step(I=current) == hand.step(current)
            assert schema.state["v"] == hand.v

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_perfect_integrator_q88_matches_hand_model_and_verilog(self) -> None:
        """Perfect Integrator has Q8.8 spike-count parity across all three paths."""
        hand_spikes = _perfect_integrator_hand_spike_count(_N_STEPS, _INPUT_CURRENT)
        schema_spikes = _python_spike_count("perfect_integrator", _N_STEPS, _INPUT_CURRENT)
        verilog_spikes = _verilog_spike_count("perfect_integrator", _N_STEPS, _INPUT_CURRENT)

        assert hand_spikes == schema_spikes == verilog_spikes == 66

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_perfect_integrator_generated_q88_declares_fractional_boundary(self) -> None:
        """Record generic-Q8.8 quantisation separately from curated source RTL."""
        current = 0.333
        hand_spikes = _perfect_integrator_hand_spike_count(_N_STEPS, current)
        schema_spikes = _python_spike_count("perfect_integrator", _N_STEPS, current)
        verilog_spikes = _verilog_spike_count("perfect_integrator", _N_STEPS, current)

        assert (hand_spikes, schema_spikes, verilog_spikes) == (32, 32, 30)

    @pytest.mark.skipif(not HAS_IVERILOG, reason="Icarus Verilog not available")
    def test_source_strict_boundary_matches_curated_q88_rtl(self) -> None:
        """Keep exact equality non-spiking in the tracked source RTL."""
        module = "sc_perfect_integrator_naud_gerstner_2012"
        schema = UniversalNeuron.from_schema("perfect_integrator")
        testbench = generate_testbench(
            schema.to_equation_neuron(),
            module_name=module,
            n_steps=1_000,
            input_current=5.0,
            data_width=16,
            fraction=8,
        )
        rtl = (ROOT / f"hdl/formal/catalogue/{module}.v").read_text(encoding="utf-8")
        source = PerfectIntegratorNeuron.naud_gerstner_2012()
        hand_events = sum(source.step(5.0) for _ in range(1_000))
        assert hand_events == simulate(rtl, testbench, module) == 333

    def test_retained_sc_schema_keeps_inclusive_equality_boundary(self) -> None:
        """Prove the source correction did not erase the historical SC model."""
        source = UniversalNeuron.from_schema("perfect_integrator")
        retained = UniversalNeuron.from_schema("sc_perfect_integrator")
        assert [source.step(I=5.0) for _ in range(3)] == [0, 0, 1]
        assert [retained.step(I=5.0) for _ in range(3)] == [0, 1, 0]
