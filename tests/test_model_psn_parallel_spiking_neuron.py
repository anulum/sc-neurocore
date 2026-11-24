# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestParallelSpikingNeuron from former test_model_psn.py

"""Focused suite: TestParallelSpikingNeuron from former test_model_psn.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.sc_resetting_psn import (
    SCResettingParallelSpikingNeuron,
)


class TestParallelSpikingNeuron:
    """Dynamics of the resetting parallel spiking neuron."""

    def test_dynamics(self) -> None:
        """The state evolves under drive."""
        n = SCResettingParallelSpikingNeuron()
        results = [n.step(0.3) for _ in range(20)]
        assert any(r != 0 for r in results) or any(b != 0.0 for b in n.buffer)
