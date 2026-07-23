# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestChialvoIsolation from former test_model_chialvo_map.py

"""Focused suite: TestChialvoIsolation from former test_model_chialvo_map.py."""

from __future__ import annotations

from tests.model_chialvo_map_support import *  # noqa: F403

class TestChialvoIsolation:
    def test_construction(self):
        n = ChialvoMapNeuron()
        assert n.x == 0.0
        assert n.y == 0.0

    def test_step_returns_binary(self):
        n = ChialvoMapNeuron()
        assert n.step(0.0) in (0, 1)

    def test_intrinsic_spiking(self):
        """Model spikes without input (k=0.04 provides excitability)."""
        n = ChialvoMapNeuron()
        spikes = sum(n.step(0.0) for _ in range(5000))
        assert spikes > 0, "no intrinsic spiking"

    def test_state_finite(self):
        n = ChialvoMapNeuron()
        for _ in range(10000):
            n.step(0.02)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_safe_exp_prevents_overflow(self):
        """Extreme y-x should not cause overflow (safe_exp used)."""
        n = ChialvoMapNeuron()
        n.y = 1000.0
        n.x = 0.0
        result = n.step(0.0)
        assert result in (0, 1)
        assert np.isfinite(n.x)
        assert np.isfinite(n.y)

    def test_reset(self):
        n = ChialvoMapNeuron()
        for _ in range(100):
            n.step(0.02)
        n.reset()
        assert n.x == 0.0
        assert n.y == 0.0

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("x", np.nan),
            ("y", np.inf),
            ("a", np.nan),
            ("b", np.inf),
            ("c", np.nan),
            ("k", np.inf),
            ("x_threshold", np.nan),
        ],
    )
    def test_rejects_invalid_numerical_configuration(self, field: str, value: float):
        with pytest.raises(ValueError):
            ChialvoMapNeuron(**{field: value})

    def test_rejects_non_finite_current_before_state_mutation(self):
        n = ChialvoMapNeuron()
        before = (n.x, n.y)
        with pytest.raises(ValueError, match="current"):
            n.step(np.nan)
        assert (n.x, n.y) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self):
        n = ChialvoMapNeuron()
        n.y = np.inf
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="state"):
            n.step(0.0)
        assert (n.x, n.y) == before

    def test_rejects_quadratic_overflow_before_state_mutation(self):
        n = ChialvoMapNeuron(x=1.0e308, y=0.0)
        before = (n.x, n.y)
        with pytest.raises(FloatingPointError, match="quadratic|candidate"):
            n.step(0.0)
        assert (n.x, n.y) == before
