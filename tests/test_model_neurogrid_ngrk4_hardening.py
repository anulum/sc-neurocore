# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNGRK4Hardening from former test_model_neurogrid.py

"""Focused suite: TestNGRK4Hardening from former test_model_neurogrid.py."""

from __future__ import annotations

from typing import Any, cast
import numpy as np
import pytest
from sc_neurocore.neurons.models.neurogrid import (
    NeuroGridNeuron,
)


class TestNGRK4Hardening:
    """The RK4 integrator, its Euler regression path and input checks of the NeuroGridNeuron."""

    def test_default_integrator_is_rk4(self) -> None:
        """Default integrator is RK4."""
        n = NeuroGridNeuron()
        assert n.integrator == "rk4"

    def test_unknown_integrator_rejected(self) -> None:
        """Unknown integrator rejected."""
        with pytest.raises(ValueError, match="Unsupported integrator"):
            NeuroGridNeuron(integrator=cast(Any, "bad"))

    def test_rk4_and_euler_regression_paths_diverge(self) -> None:
        """RK4 and Euler regression paths diverge."""
        rk4 = NeuroGridNeuron()
        euler = NeuroGridNeuron(integrator="baseline_euler")
        rk4_spikes = sum(rk4.step(100.0) for _ in range(20_000))
        euler_spikes = sum(euler.step(100.0) for _ in range(20_000))
        assert rk4_spikes == 94
        assert euler_spikes == 93

    def test_cross_backend_anchor(self) -> None:
        """Cross-backend anchor."""
        n = NeuroGridNeuron()
        spikes = sum(n.step(100.0) for _ in range(20_000))
        assert spikes == 94
        assert np.isfinite(n.v_s) and np.isfinite(n.v_d)

    def test_invalid_input_preserves_state(self) -> None:
        """Invalid input preserves state."""
        n = NeuroGridNeuron()
        for _ in range(10):
            n.step(100.0)
        old_state = (n.v_s, n.v_d)
        with pytest.raises(ValueError, match="current must be finite"):
            n.step(float("nan"))
        assert (n.v_s, n.v_d) == old_state

    def test_corrupt_state_rejected_before_mutation(self) -> None:
        """Corrupt state rejected before mutation."""
        n = NeuroGridNeuron()
        for _ in range(10):
            n.step(100.0)
        old_v_d = n.v_d
        n.v_s = float("nan")
        with pytest.raises(ValueError, match="v_s must be finite"):
            n.step(100.0)
        assert n.v_d == old_v_d

    def test_runtime_configuration_rejects_invalid_tau(self) -> None:
        """Runtime configuration rejects invalid tau."""
        n = NeuroGridNeuron()
        n.tau_s = 0.0
        with pytest.raises(ValueError, match="tau_s must be positive"):
            n.step(100.0)
