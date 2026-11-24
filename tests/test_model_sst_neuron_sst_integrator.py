# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTIntegrator from former test_model_sst_neuron.py

"""Focused suite: TestSSTIntegrator from former test_model_sst_neuron.py."""

from __future__ import annotations

from typing import Any, cast
import pytest
from sc_neurocore.neurons.models.sst_neuron import (
    SSTNeuron,
)
from tests.model_sst_neuron_support import (
    _spikes,
)


class TestSSTIntegrator:
    """Integrator choice and agreement of the SSTNeuron."""

    def test_default_integrator_is_rk4(self) -> None:
        """Default integrator is RK4."""
        assert SSTNeuron().integrator == "rk4"

    def test_rejects_unknown_integrator(self) -> None:
        """The model rejects unknown integrator."""
        with pytest.raises(ValueError, match="Unsupported integrator"):
            SSTNeuron(integrator=cast(Any, "midpoint"))

    def test_baseline_euler_path_runs_and_diverges_from_rk4(self) -> None:
        """Baseline Euler path runs and diverges from RK4."""
        rk4 = SSTNeuron()
        euler = SSTNeuron(integrator="baseline_euler")
        assert _spikes(rk4, 5.0, 40000) > 0
        assert _spikes(euler, 5.0, 40000) > 0
        assert rk4.v != euler.v

    def test_rk4_and_euler_agree_to_first_order_at_tiny_dt(self) -> None:
        """RK4 and Euler agree to first order at tiny dt."""
        rk4 = SSTNeuron(dt=1e-5)
        euler = SSTNeuron(dt=1e-5, integrator="baseline_euler")
        for _ in range(200):
            rk4.step(2.0)
            euler.step(2.0)
        assert abs(rk4.v - euler.v) < 1e-2
