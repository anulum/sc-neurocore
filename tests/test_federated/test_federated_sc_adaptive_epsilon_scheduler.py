# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAdaptiveEpsilonScheduler from former test_federated_sc.py

"""Focused suite: TestAdaptiveEpsilonScheduler from former test_federated_sc.py."""

from __future__ import annotations

from sc_neurocore.federated.federated_sc import (
    AdaptiveEpsilonScheduler,
)


class TestAdaptiveEpsilonScheduler:
    """Scheduling the privacy budget per round."""

    def test_initial_epsilon(self) -> None:
        """The schedule starts at the base epsilon."""
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0)
        assert sched.current_epsilon == 2.0

    def test_decay_on_convergence(self) -> None:
        """A converging step halves epsilon from 2 to 1."""
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        eps = sched.step(converging=True)
        assert eps == 1.0

    def test_increase_on_divergence(self) -> None:
        """A diverging step doubles epsilon from 0.5 to 1."""
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        sched.current_epsilon = 0.5
        eps = sched.step(converging=False)
        assert eps == 1.0

    def test_min_epsilon_floor(self) -> None:
        """A steep decay stops at the minimum epsilon."""
        sched = AdaptiveEpsilonScheduler(base_epsilon=1.0, decay_rate=0.01, min_epsilon=0.5)
        eps = sched.step(converging=True)
        assert eps >= 0.5

    def test_max_epsilon_cap(self) -> None:
        """Growth stops at the base epsilon."""
        sched = AdaptiveEpsilonScheduler(base_epsilon=2.0, decay_rate=0.5)
        sched.current_epsilon = 1.5
        eps = sched.step(converging=False)
        assert eps <= 2.0
