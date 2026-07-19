# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hybrid linear-attention model contracts

"""Module-specific behavioural contracts for ``HybridLinearAttentionNeuron``."""

from __future__ import annotations

import math

import pytest


class TestHybridLinearAttentionNeuron:
    @pytest.fixture()
    def neuron(self):
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        return HybridLinearAttentionNeuron(dim=16)

    def test_defaults(self, neuron):
        assert neuron.dim == 16
        assert neuron.lambda_decay == 0.95
        assert neuron.window_size == 16

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"dim": 0},
            {"dim": 1.5},
            {"lambda_decay": -0.01},
            {"lambda_decay": 1.01},
            {"lambda_decay": float("nan")},
            {"window_size": 0},
            {"window_size": 2.5},
            {"dt": 0.0},
            {"v": float("inf")},
            {"_state_kv": [0.0, float("nan")]},
            {"_window_buf": [0.0, float("inf")]},
        ],
    )
    def test_rejects_non_physical_attention_parameters(self, kwargs):
        """Hybrid attention state must be finite, bounded, and dimensionally valid."""
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        with pytest.raises(ValueError):
            HybridLinearAttentionNeuron(**kwargs)

    @pytest.mark.parametrize(
        ("query", "key", "value"),
        [(float("nan"), 0.0, 0.0), (0.0, float("inf"), 0.0), (0.0, 0.0, float("nan"))],
    )
    def test_rejects_non_finite_qkv_drive(self, query, key, value):
        """Attention update must fail closed on non-finite projections."""
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        with pytest.raises(ValueError, match="query, key, and value"):
            HybridLinearAttentionNeuron().step_qkv(query, key, value)

    def test_step_qkv_returns_float(self, neuron):
        out = neuron.step_qkv(1.0, 0.5, 2.0)
        assert isinstance(out, float)

    def test_step_returns_binary(self, neuron):
        s = neuron.step(0.5)
        assert s in (0, 1)

    def test_phi_feature_map(self, neuron):
        """phi(x) = elu(x) + 1: positive -> x+1, negative -> exp(x)."""
        assert neuron._phi(2.0) == 3.0
        assert abs(neuron._phi(-1.0) - math.exp(-1.0)) < 1e-10
        assert neuron._phi(0.0) == 1.0  # boundary

    def test_recurrent_state_decays(self, neuron):
        """Lambda decay: state_kv *= lambda each step."""
        neuron.step_qkv(1.0, 1.0, 10.0)
        first_v = neuron.v
        # Feed zeros — state decays.
        for _ in range(50):
            neuron.step_qkv(0.01, 0.01, 0.0)
        assert abs(neuron.v) < abs(first_v)

    def test_window_buffer_averaging(self, neuron):
        """Local attention = sliding window average of values."""
        for i in range(16):
            neuron.step_qkv(0.0, 0.0, float(i))
        # Window now has [0..15], mean = 7.5.
        # With q=0 → phi(0) = 1, global component is small.
        # local = mean(window) = 7.5, v ≈ 0.5 * global + 0.5 * 7.5

    def test_reset(self, neuron):
        for _ in range(20):
            neuron.step(2.0)
        neuron.reset()
        assert neuron.v == 0.0
        assert all(s == 0.0 for s in neuron._state_kv)
        assert all(w == 0.0 for w in neuron._window_buf)

    def test_different_dims(self):
        from sc_neurocore.neurons.models import HybridLinearAttentionNeuron

        for dim in [4, 32, 64]:
            n = HybridLinearAttentionNeuron(dim=dim)
            assert len(n._state_kv) == dim
            n.step_qkv(1.0, 1.0, 1.0)
