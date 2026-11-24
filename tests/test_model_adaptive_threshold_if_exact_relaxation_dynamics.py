# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExactRelaxationDynamics from former test_model_adaptive_threshold_if.py

"""Focused suite: TestExactRelaxationDynamics from former test_model_adaptive_threshold_if.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.adaptive_threshold_if import (
    AdaptiveThresholdIFNeuron,
)


class TestExactRelaxationDynamics:
    """Each step is the exact constant-input relaxation, never an Euler step."""

    def test_subthreshold_step_matches_exact_relaxation(self) -> None:
        """A subthreshold step follows the exact relaxation."""
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0, dt=0.25)
        expected_v = n.v_rest + 12.0 + (n.v - (n.v_rest + 12.0)) * np.exp(-n.dt / n.tau_m)
        expected_theta = n.theta_rest + (n.theta - n.theta_rest) * np.exp(-n.dt / n.tau_theta)

        assert n.step(12.0) == 0

        assert n.v == pytest.approx(expected_v, rel=1e-14, abs=1e-14)
        assert n.theta == pytest.approx(expected_theta, rel=1e-14, abs=1e-14)

    def test_step_is_not_forward_euler(self) -> None:
        """The step is not a forward-Euler step."""
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0, dt=0.5)
        n.step(12.0)
        euler_v = -70.0 + (-(-70.0 - (-65.0)) + 12.0) / 10.0 * 0.5
        assert abs(n.v - euler_v) > 1.0e-3

    def test_large_timestep_exact_relaxation_remains_bounded(self) -> None:
        """A large time step stays bounded."""
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-30.0, tau_m=0.04, tau_theta=0.04, dt=1.0)

        assert n.step(0.0) == 0

        assert n.v == pytest.approx(n.v_rest, rel=0.0, abs=1e-8)
        assert n.theta == pytest.approx(n.theta_rest, rel=0.0, abs=1e-8)

    def test_subthreshold_relaxation_is_monotone_toward_rest(self) -> None:
        """Subthreshold relaxation moves steadily toward rest."""
        n = AdaptiveThresholdIFNeuron(v=-70.0, theta=-40.0)
        v_before = n.v
        theta_before = n.theta

        assert n.step(0.0) == 0

        assert v_before < n.v < n.v_rest
        assert n.theta_rest < n.theta < theta_before

    def test_steady_state_is_a_fixed_point(self) -> None:
        """The steady state is a fixed point."""
        n = AdaptiveThresholdIFNeuron(v=-53.0, theta=-50.0)
        n.v = n.v_rest + 12.0
        n.theta = n.theta_rest
        assert n.step(12.0) == 0
        assert n.v == pytest.approx(n.v_rest + 12.0, rel=0.0, abs=1e-14)
        assert n.theta == pytest.approx(n.theta_rest, rel=0.0, abs=1e-14)
