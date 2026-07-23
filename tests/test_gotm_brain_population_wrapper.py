# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPopulationWrapper from former test_gotm_brain.py

"""Focused suite: TestPopulationWrapper from former test_gotm_brain.py."""

from __future__ import annotations

from tests.gotm_brain_support import *  # noqa: F403

class TestPopulationWrapper:
    def test_step_returns_int(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        result = n.step(0.0)
        assert isinstance(result, int)
        assert result in (0, 1)

    def test_v_property(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        assert n.v == -70.0
        n.v = -55.0
        assert n.v == -55.0

    def test_spiking(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        total_spikes = 0
        for _ in range(200):
            total_spikes += n.step(50.0)
        assert total_spikes > 0

    def test_reset(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        n.step(50.0)
        n.reset()
        assert n.v == -70.0

    def test_get_state(self) -> None:
        from sc_neurocore.quantum_cognition.fisher_posner import (
            HybridFisherPosnerLIFNeuron,
        )

        HybridFisherPosnerLIFNeuron._reset_pools()
        n = HybridFisherPosnerLIFNeuron()
        state = n.get_state()
        assert "Vm" in state
        assert "atp_level" in state
