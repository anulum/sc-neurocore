# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOnlineTrainer from former test_online_learning.py

"""Focused suite: TestOnlineTrainer from former test_online_learning.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.online_learning.online_trainer import (
    OnlineTrainer,
)


class TestOnlineTrainer:
    """The multi-layer online trainer."""

    def test_init(self) -> None:
        """The trainer builds its layers."""
        ot = OnlineTrainer(layer_sizes=[10, 20, 5])
        assert ot.n_layers == 2
        assert ot.layers[0].n_inputs == 10
        assert ot.layers[0].n_neurons == 20
        assert ot.layers[1].n_inputs == 20
        assert ot.layers[1].n_neurons == 5

    def test_step_no_learning(self) -> None:
        """A step without learning reports no loss."""
        ot = OnlineTrainer(layer_sizes=[4, 8, 2])
        result = ot.step(np.random.rand(4))
        assert result["output"].shape == (2,)
        assert "loss" not in result

    def test_step_with_learning(self) -> None:
        """A learning step reports a loss."""
        ot = OnlineTrainer(layer_sizes=[4, 8, 2], lr=0.01)
        result = ot.step(np.random.rand(4), target=np.array([1.0, 0.0]))
        assert "loss" in result
        assert result["loss"] >= 0

    def test_train_sequence(self) -> None:
        """Training on a sequence gives a loss."""
        ot = OnlineTrainer(layer_sizes=[4, 16, 2], lr=0.001)
        inputs = np.random.rand(30, 4)
        targets = np.zeros((30, 2))
        loss = ot.train_sequence(inputs, targets)
        assert loss >= 0

    def test_reset(self) -> None:
        """A reset zeroes the voltages."""
        ot = OnlineTrainer(layer_sizes=[4, 8, 2])
        ot.step(np.ones(4))
        ot.reset()
        for layer in ot.layers:
            assert np.allclose(layer._v, 0)

    def test_memory_per_step(self) -> None:
        """The memory per step is positive."""
        ot = OnlineTrainer(layer_sizes=[10, 20, 5])
        assert ot.memory_per_step > 0

    def test_three_layers(self) -> None:
        """Three layers give the asked output size."""
        ot = OnlineTrainer(layer_sizes=[8, 16, 8, 4])
        assert ot.n_layers == 3
        result = ot.step(np.random.rand(8))
        assert result["output"].shape == (4,)
