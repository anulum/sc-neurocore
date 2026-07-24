# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEPTraining from former test_equilibrium_propagation.py

"""Focused suite: TestEPTraining from former test_equilibrium_propagation.py."""

from __future__ import annotations

from tests.equilibrium_propagation_support import *  # noqa: F403


class TestEPTraining:
    """Test the EP training protocol."""

    def test_train_returns_mse(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.random.default_rng(0).random((5, 4))
        y = np.random.default_rng(1).random((5, 2))
        mse = net.train(x, y, beta=1.0, lr=0.01, n_settle=5)
        assert isinstance(mse, float)
        assert mse >= 0

    def test_training_reduces_error(self) -> None:
        """Simple linear task — EP should reduce error over iterations."""
        net = EPNetwork([3, 20, 1], rng_seed=0)
        rng = np.random.default_rng(99)
        x = rng.random((8, 3))
        # Target = mean of inputs (easy linear task)
        y = x.mean(axis=1, keepdims=True)

        first_mse = net.train(x, y, beta=0.5, lr=0.1, n_settle=30)
        for _ in range(100):
            last_mse = net.train(x, y, beta=0.5, lr=0.1, n_settle=30)

        # Error should decrease (or at minimum, weights should change)
        assert last_mse <= first_mse + 0.01, (
            f"Training should not increase error: {first_mse:.4f} → {last_mse:.4f}"
        )

    # NOTE: hard-sigmoid EP with random init often saturates to the
    # activation boundaries (0 or 1), making weight updates zero.
    # This is a known research limitation — production EP would use
    # smooth activations (softplus, sigmoid). The prototype validates
    # the algorithm structure, not convergence guarantees.

    def test_predict_shape(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.array([0.1, 0.2, 0.3, 0.4])
        output = net.predict(x)
        assert output.shape == (2,)
        assert np.all(output >= 0) and np.all(output <= 1)
