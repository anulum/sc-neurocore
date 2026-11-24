# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEnergyLIFValidation from former test_model_energy_lif.py

"""Focused suite: TestEnergyLIFValidation from former test_model_energy_lif.py."""

from __future__ import annotations

from typing import Any
import numpy as np
import pytest
from sc_neurocore.neurons.models.sc_normalized_energy_lif import (
    SCNormalizedEnergyLIFNeuron as EnergyLIFNeuron,
)


class TestEnergyLIFValidation:
    """Input and state validation of the EnergyLIFNeuron."""

    @pytest.mark.parametrize("field", ["v", "v_rest", "v_reset", "v_threshold"])
    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_voltage_parameters(self, field: str, value: float) -> None:
        """The model rejects non-finite voltage parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            EnergyLIFNeuron(**kwargs)

    @pytest.mark.parametrize("field", ["epsilon", "epsilon_0"])
    @pytest.mark.parametrize("value", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_energy_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects negative or non-finite energy parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            EnergyLIFNeuron(**kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"epsilon": 1.1},
            {"epsilon": 0.2, "epsilon_0": 0.1},
            {"v_threshold": -75.0},
            {"v_reset": -45.0},
            {"dt": 11.0},
            {"dt": 501.0},
        ],
    )
    def test_rejects_non_physical_energy_geometry_or_timestep(self, kwargs: dict[str, Any]) -> None:
        """The model rejects non-physical energy geometry or timestep."""
        with pytest.raises(ValueError):
            EnergyLIFNeuron(**kwargs)

    def test_energy_recovery_is_monotone_and_bounded_without_spike(self) -> None:
        """Energy recovery is monotone and bounded without spike."""
        n = EnergyLIFNeuron(epsilon=0.2)
        before = n.epsilon

        assert n.step(0.0) == 0

        assert before < n.epsilon < n.epsilon_0

    def test_exact_candidate_commit(self) -> None:
        """The exact-flow candidate is committed as computed."""
        n = EnergyLIFNeuron(epsilon=0.5)
        expected_v, expected_epsilon = n._exact_candidate(10.0)

        assert n.step(10.0) == 0

        assert abs(n.v - expected_v) < 1.0e-12
        assert abs(n.epsilon - expected_epsilon) < 1.0e-12

    def test_exact_flow_separates_from_forward_euler(self) -> None:
        """Exact flow separates from forward Euler."""
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5, dt=2.0)
        euler_v = n.v + (-(n.v - n.v_rest) + n.resistance * n.epsilon * 10.0) / n.tau_m * n.dt
        exact_v, _ = n._exact_candidate(10.0)

        assert abs(exact_v - euler_v) > 1.0e-3

    def test_spike_uses_energy_candidate(self) -> None:
        """Spike uses energy candidate."""
        n = EnergyLIFNeuron()
        _, epsilon_candidate = n._exact_candidate(250.0)

        assert n.step(250.0) == 1

        assert n.v == n.v_reset
        assert abs(n.epsilon - max(0.0, epsilon_candidate - n.alpha)) < 1.0e-12

    @pytest.mark.parametrize("field", ["tau_m", "tau_e", "resistance", "dt"])
    @pytest.mark.parametrize("value", [0.0, -1.0, np.nan, np.inf])
    def test_rejects_non_positive_or_non_finite_scale_parameters(
        self, field: str, value: float
    ) -> None:
        """The model rejects non-positive or non-finite scale parameters."""
        with pytest.raises(ValueError, match=field):
            kwargs: dict[str, Any] = {field: value}
            EnergyLIFNeuron(**kwargs)

    @pytest.mark.parametrize("alpha", [-1.0, np.nan, np.inf, -np.inf])
    def test_rejects_negative_or_non_finite_spike_cost(self, alpha: float) -> None:
        """The model rejects negative or non-finite spike cost."""
        with pytest.raises(ValueError, match="alpha"):
            EnergyLIFNeuron(alpha=alpha)

    @pytest.mark.parametrize("current", [np.nan, np.inf, -np.inf])
    def test_rejects_non_finite_current_before_state_mutation(self, current: float) -> None:
        """A non-finite current is rejected before any state change."""
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5)
        before = (n.v, n.epsilon)
        with pytest.raises(ValueError, match="current"):
            n.step(current)
        assert (n.v, n.epsilon) == before

    def test_rejects_corrupted_runtime_state_before_mutation(self) -> None:
        """Corrupted runtime state is rejected before any change."""
        n = EnergyLIFNeuron(v=-65.0, epsilon=0.5)
        n.epsilon = -1.0
        before = (n.v, n.epsilon)

        with pytest.raises(ValueError, match="epsilon"):
            n.step(10.0)

        assert (n.v, n.epsilon) == before
