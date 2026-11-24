# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMartinottiIntegrator from former test_model_martinotti_neuron.py

"""Focused suite: TestMartinottiIntegrator from former test_model_martinotti_neuron.py."""

from __future__ import annotations

from typing import Any, cast
import pytest
from sc_neurocore.neurons.models.martinotti_neuron import (
    MartinottiNeuron,
)
from tests.model_martinotti_neuron_support import (
    _spikes,
)


class TestMartinottiIntegrator:
    """Integrator choice and agreement of the MartinottiNeuron."""

    def test_default_integrator_is_rk4(self) -> None:
        """Default integrator is RK4."""
        assert MartinottiNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self) -> None:
        """The model rejects unknown integrator."""
        with pytest.raises(ValueError, match="Unsupported integrator"):
            MartinottiNeuron(integrator=cast(Any, "midpoint"))

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self) -> None:
        """Baseline Euler path runs and diverges from RK4."""
        rk4 = MartinottiNeuron()
        euler = MartinottiNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 5.0, 40000) > 0
        assert _spikes(euler, 5.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self) -> None:
        """RK4 and Euler agree to first order at tiny dt."""
        rk4 = MartinottiNeuron(dt=1e-5)
        euler = MartinottiNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(5.0)
            euler.step(5.0)
        assert abs(rk4.v - euler.v) < 1e-2
