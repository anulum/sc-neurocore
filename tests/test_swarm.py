# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for robotics swarm coupling module

"""Real-surface tests for robotics swarm coupling."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.layers.sc_learning_layer import SCLearningLayer
from sc_neurocore.robotics.swarm import SwarmCoupling


class TestSwarmCoupling:
    """SwarmCoupling public behavior checks."""

    def test_construction(self) -> None:
        """Construction preserves the configured coupling strength."""
        sc = SwarmCoupling(coupling_strength=0.2)
        assert sc.coupling_strength == 0.2

    def test_synchronize_shifts_weights(self) -> None:
        """Synchronization mutates both agents through the public layer API."""
        a = SCLearningLayer(n_inputs=4, n_neurons=3)
        b = SCLearningLayer(n_inputs=4, n_neurons=3)
        wa_before = a.get_weights().copy()
        wb_before = b.get_weights().copy()

        sc = SwarmCoupling(coupling_strength=0.5)
        sc.synchronize(a, b)

        wa_after = a.get_weights()
        wb_after = b.get_weights()
        assert not np.array_equal(wa_before, wa_after)
        assert not np.array_equal(wb_before, wb_after)

    def test_synchronize_converges(self) -> None:
        """Repeated synchronization reduces inter-agent weight distance."""
        a = SCLearningLayer(n_inputs=4, n_neurons=3)
        b = SCLearningLayer(n_inputs=4, n_neurons=3)

        sc = SwarmCoupling(coupling_strength=0.3)
        for _ in range(20):
            sc.synchronize(a, b)

        wa = a.get_weights()
        wb = b.get_weights()
        diff = np.abs(wa - wb).mean()
        assert diff < 0.1

    def test_mismatched_sizes_raises(self) -> None:
        """Agents with different neuron counts fail closed."""
        a = SCLearningLayer(n_inputs=4, n_neurons=3)
        b = SCLearningLayer(n_inputs=4, n_neurons=5)
        sc = SwarmCoupling()
        with pytest.raises(ValueError, match="same size"):
            sc.synchronize(a, b)
