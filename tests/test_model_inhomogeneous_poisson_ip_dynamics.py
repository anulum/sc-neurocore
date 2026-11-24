# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPDynamics from former test_model_inhomogeneous_poisson.py

"""Focused suite: TestIPDynamics from former test_model_inhomogeneous_poisson.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.inhomogeneous_poisson import (
    InhomogeneousPoissonNeuron,
)
from tests.model_inhomogeneous_poisson_support import (
    _run,
)


class TestIPDynamics:
    """Dynamics of the InhomogeneousPoissonNeuron."""

    def test_fires_at_positive_rate(self) -> None:
        """The model fires at positive rate."""
        n = InhomogeneousPoissonNeuron()
        spikes = _run(n, rate=100.0, steps=10_000)
        assert len(spikes) >= 100

    def test_rate_monotonic(self) -> None:
        """The firing rate does not fall as the current rises."""
        s_low = len(_run(InhomogeneousPoissonNeuron(), 50.0, 10_000))
        s_high = len(_run(InhomogeneousPoissonNeuron(), 500.0, 10_000))
        assert s_high >= s_low
