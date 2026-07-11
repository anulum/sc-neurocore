# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Perfect Integrator co-simulation contracts

"""Perfect Integrator schema, hand-model, and RTL parity contracts."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from tests.cosim_support import (
    HAS_IVERILOG,
    _perfect_integrator_hand_spike_count,
    _python_spike_count,
    _verilog_spike_count,
)

_N_STEPS = 200
_INPUT_CURRENT = 50.0


class TestTierBModelCosim:
    """WC-A5 Tier-B Perfect Integrator enrolment."""

    def test_perfect_integrator_schema_matches_hand_model_sequence(self) -> None:
        """The schema mirrors the hand-authored non-leaky integrator step law."""
        hand = PerfectIntegratorNeuron()
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

        assert hand_spikes == schema_spikes == verilog_spikes == _N_STEPS
