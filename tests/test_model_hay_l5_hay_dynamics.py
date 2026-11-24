# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestHayDynamics from former test_model_hay_l5.py

"""Focused suite: TestHayDynamics from former test_model_hay_l5.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.hay_l5 import (
    HayL5PyramidalNeuron,
)
from tests.model_hay_l5_support import (
    _run,
)


class TestHayDynamics:
    """Dynamics of the HayL5PyramidalNeuron."""

    def test_fires_under_somatic_drive(self) -> None:
        """The model fires under somatic drive."""
        n = HayL5PyramidalNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 1

    def test_subthreshold_silent(self) -> None:
        """A subthreshold current gives no spikes."""
        n = HayL5PyramidalNeuron()
        assert len(_run(n, current=1.0, steps=3000)) == 0

    def test_rate_monotonic(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [5.0, 10.0, 20.0]:
            n = HayL5PyramidalNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]
