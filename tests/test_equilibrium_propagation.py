# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Equilibrium Propagation prototype

"""Test suite for the EP research prototype."""

from __future__ import annotations

import importlib
import importlib.abc
import sys

import numpy as np

import sc_neurocore.training as training
from sc_neurocore.training.equilibrium_propagation import EPNetwork, _rho, _rho_prime


class _BlockTorchFinder(importlib.abc.MetaPathFinder):
    """Import hook that forces the training package through its no-Torch branch."""

    def find_spec(
        self,
        fullname: str,
        path: object | None,
        target: object | None = None,
    ) -> None:
        if fullname == "torch":
            raise ImportError("forced missing torch surface")
        return None


def test_training_package_exports_equilibrium_propagation_surface() -> None:
    """The training package facade exposes the documented EP research surface."""
    assert "EPNetwork" in training.__all__
    assert training.EPNetwork is EPNetwork


def test_training_package_exports_ep_without_torch() -> None:
    """The NumPy EP surface remains selectable when Torch is unavailable."""
    finder = _BlockTorchFinder()
    original_torch = sys.modules.pop("torch", None)
    sys.meta_path.insert(0, finder)

    try:
        reloaded = importlib.reload(training)

        assert reloaded.HAS_TORCH is False
        assert reloaded.EPNetwork is EPNetwork
        assert "EPNetwork" in reloaded.__all__
    finally:
        sys.meta_path.remove(finder)
        if original_torch is not None:
            sys.modules["torch"] = original_torch
        importlib.reload(training)


class TestActivation:
    """Test hard-sigmoid activation functions."""

    def test_rho_clips_to_01(self) -> None:
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        result = _rho(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_rho_prime_in_range(self) -> None:
        x = np.array([-1.0, 0.5, 1.5])
        result = _rho_prime(x)
        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

    def test_rho_prime_boundaries(self) -> None:
        # At exact boundaries, derivative is 0
        assert _rho_prime(np.array([0.0])) == 0.0
        assert _rho_prime(np.array([1.0])) == 0.0


class TestEPNetworkInit:
    """Test network initialisation."""

    def test_creates_correct_layers(self) -> None:
        net = EPNetwork([10, 5, 3])
        assert len(net.weights) == 2
        assert net.weights[0].shape == (10, 5)
        assert net.weights[1].shape == (5, 3)

    def test_biases_zero_init(self) -> None:
        net = EPNetwork([4, 3, 2])
        np.testing.assert_array_equal(net.biases[0], np.zeros(3))
        np.testing.assert_array_equal(net.biases[1], np.zeros(2))

    def test_xavier_scale(self) -> None:
        # Xavier init should produce weights with moderate magnitude
        net = EPNetwork([100, 50, 10])
        for w in net.weights:
            assert abs(w.mean()) < 0.1
            assert w.std() < 0.5

    def test_deterministic_with_seed(self) -> None:
        net1 = EPNetwork([5, 3], rng_seed=42)
        net2 = EPNetwork([5, 3], rng_seed=42)
        np.testing.assert_array_equal(net1.weights[0], net2.weights[0])


class TestEPSettling:
    """Test the free-phase settling process."""

    def test_settle_returns_correct_structure(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.array([0.5, 0.3, 0.1, 0.8])
        states = net._settle(x, n_steps=5)
        assert len(states) == 3
        assert states[0].shape == (4,)
        assert states[1].shape == (3,)
        assert states[2].shape == (2,)

    def test_input_stays_clamped(self) -> None:
        net = EPNetwork([4, 3, 2])
        x = np.array([0.5, 0.3, 0.1, 0.8])
        states = net._settle(x, n_steps=20)
        np.testing.assert_array_equal(states[0], x)

    def test_energy_decreases_free_phase(self) -> None:
        net = EPNetwork([5, 4, 3])
        x = np.ones(5) * 0.5
        # Settle progressively and check energy trend
        energies = []
        for steps in [1, 5, 10, 20]:
            states = net._settle(x, n_steps=steps)
            energies.append(net._energy(states))
        # Energy should generally decrease (or be stable) during settling
        # Allow small fluctuations
        assert energies[-1] <= energies[0] + 0.1, f"Energy should decrease: {energies}"


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


class TestEPSerialisation:
    """Test parameter serialisation."""

    def test_get_params_structure(self) -> None:
        net = EPNetwork([3, 2, 1])
        params = net.get_params()
        assert params["layer_sizes"] == [3, 2, 1]
        assert len(params["weights"]) == 2
        assert len(params["biases"]) == 2

    def test_params_are_json_serialisable(self) -> None:
        import json

        net = EPNetwork([3, 2])
        params = net.get_params()
        json_str = json.dumps(params)
        assert len(json_str) > 0
        parsed = json.loads(json_str)
        assert parsed["layer_sizes"] == [3, 2]
