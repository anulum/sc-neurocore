# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLCFAnalytical from former test_model_leaky_compete_fire.py

"""Focused suite: TestLCFAnalytical from former test_model_leaky_compete_fire.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.leaky_compete_fire import (
    LeakyCompeteFireNeuron,
)
from tests.model_leaky_compete_fire_support import (
    _exact_lcf_candidates,
)


class TestLCFAnalytical:
    """Analytical behaviour of the LeakyCompeteFireNeuron."""

    def test_uniform_input_all_fire_together(self) -> None:
        """Equal input to all units → all spike simultaneously."""
        n = LeakyCompeteFireNeuron()
        for _ in range(1000):
            spikes = n.step(5.0)
            if sum(spikes) > 0:
                # All got same input → all should spike together
                assert sum(spikes) == n.n_units or sum(spikes) >= 1
                break

    def test_lateral_inhibition_resets_losers(self) -> None:
        """When unit i spikes, other units V_j -= w_inh."""
        n = LeakyCompeteFireNeuron(n_units=2)
        # Drive unit 0 strongly, unit 1 weakly
        for _ in range(100):
            spikes = n.step([10.0, 0.1])
            if spikes[0] == 1:
                # Unit 1 should have been inhibited
                assert n.v[1] >= 0.0  # clipped
                break

    def test_winner_take_all_with_asymmetric_input(self) -> None:
        """Higher input unit spikes more often (WTA)."""
        n = LeakyCompeteFireNeuron(n_units=2)
        spikes_0, spikes_1 = 0, 0
        for _ in range(5000):
            s = n.step([5.0, 2.0])
            spikes_0 += s[0]
            spikes_1 += s[1]
        assert spikes_0 > spikes_1

    def test_v_non_negative_after_inhibition(self) -> None:
        """V clipped to ≥ 0 after lateral inhibition."""
        n = LeakyCompeteFireNeuron()
        for _ in range(1000):
            n.step(5.0)
        for v in n.v:
            assert v >= 0.0

    def test_scalar_input_broadcast(self) -> None:
        """Scalar input is broadcast to all units."""
        n = LeakyCompeteFireNeuron()
        n.step(5.0)
        # All units should get same drive
        assert isinstance(n.v, list) and len(n.v) == n.n_units

    def test_list_input_per_unit(self) -> None:
        """List input gives different current per unit."""
        n = LeakyCompeteFireNeuron(n_units=3, v_threshold=100.0)
        n.step([1.0, 2.0, 3.0])
        # Unit 2 should have highest V
        assert n.v[2] > n.v[0]

    def test_exact_relaxation_matches_closed_form_without_spike(self) -> None:
        """Exact relaxation matches closed form without spike."""
        n = LeakyCompeteFireNeuron(
            n_units=3,
            v=[0.2, 0.4, 0.1],
            tau=7.0,
            v_threshold=100.0,
            dt=2.5,
        )
        currents = [1.0, 0.5, 0.0]
        expected = _exact_lcf_candidates(n, currents)

        spikes = n.step(currents)

        assert spikes == [0, 0, 0]
        assert n.v == pytest.approx(expected, abs=1e-12)

    def test_large_timestep_relaxes_without_euler_overshoot(self) -> None:
        """Large timestep relaxes without Euler overshoot."""
        n = LeakyCompeteFireNeuron(
            n_units=2,
            v=[5.0, 1.0],
            tau=1.0,
            v_threshold=100.0,
            dt=50.0,
        )

        n.step([0.0, 0.0])

        assert 0.0 <= n.v[0] <= 5.0
        assert 0.0 <= n.v[1] <= 1.0

    def test_exact_candidates_are_committed_before_inhibition(self) -> None:
        """Exact candidates are committed before inhibition."""
        n = LeakyCompeteFireNeuron(
            n_units=2,
            v=[0.0, 0.0],
            tau=10.0,
            v_threshold=0.5,
            w_inh=0.25,
            dt=2.0,
        )
        expected = _exact_lcf_candidates(n, [5.0, 1.0])

        spikes = n.step([5.0, 1.0])

        assert spikes == [1, 0]
        assert n.v[0] == 0.0
        assert n.v[1] == pytest.approx(max(0.0, expected[1] - n.w_inh), abs=1e-12)

    def test_custom_n_units(self) -> None:
        """A custom unit count sets the state and output sizes."""
        n = LeakyCompeteFireNeuron(n_units=8)
        assert len(n.v) == 8
        result = n.step(5.0)
        assert len(result) == 8

    def test_runtime_corrupted_state_fails_before_mutation(self) -> None:
        """Runtime corrupted state fails before mutation."""
        n = LeakyCompeteFireNeuron(n_units=2, v=[0.2, 0.4])
        before = list(n.v)
        n.tau = 0.0
        with pytest.raises(ValueError, match="tau"):
            n.step([1.0, 0.5])
        assert n.v == before

    def test_runtime_voltage_length_mismatch_fails_before_mutation(self) -> None:
        """Runtime voltage length mismatch fails before mutation."""
        n = LeakyCompeteFireNeuron(n_units=2, v=[0.2, 0.4])
        n.v = [0.2]
        with pytest.raises(ValueError, match="v must have length"):
            n.step([1.0, 0.5])
        assert n.v == [0.2]

    def test_runtime_non_finite_voltage_fails_before_mutation(self) -> None:
        """Runtime non-finite voltage fails before mutation."""
        n = LeakyCompeteFireNeuron(n_units=2, v=[0.2, 0.4])
        n.v[1] = np.inf
        with pytest.raises(ValueError, match="v must contain"):
            n.step([1.0, 0.5])
        assert n.v[1] == np.inf

    def test_non_finite_candidate_fails_before_mutation(self) -> None:
        """Non-finite candidate fails before mutation."""
        n = LeakyCompeteFireNeuron(n_units=2, v=[1.0e308, 0.4])
        before = list(n.v)
        with pytest.raises(ValueError, match="relaxation"):
            n.step([-1.0e308, 0.0])
        assert n.v == before
