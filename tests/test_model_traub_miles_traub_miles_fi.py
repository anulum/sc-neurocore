# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTraubMilesFI from former test_model_traub_miles.py

"""Focused suite: TestTraubMilesFI from former test_model_traub_miles.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.traub_miles import (
    TraubMilesNeuron,
)
from tests.model_traub_miles_support import (
    _run,
)


class TestTraubMilesFI:
    """The current-to-rate response of the TraubMilesNeuron."""

    def test_subthreshold_silent(self) -> None:
        """A subthreshold current gives no spikes."""
        n = TraubMilesNeuron()
        assert len(_run(n, current=0.0, steps=50000)) == 0

    def test_suprathreshold_fires(self) -> None:
        """A suprathreshold current fires repeatedly."""
        n = TraubMilesNeuron()
        assert len(_run(n, current=2.0, steps=50000)) >= 100

    def test_monotonic_fi(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [1.0, 2.0, 5.0, 10.0, 20.0]:
            n = TraubMilesNeuron()
            rates.append(len(_run(n, current=I, steps=50000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_rate_scales_sublinearly(self) -> None:
        """HH f-I is not linear — verify monotonic but non-trivial scaling."""
        n2 = TraubMilesNeuron()
        n10 = TraubMilesNeuron()
        s2 = len(_run(n2, current=2.0, steps=50000))
        s10 = len(_run(n10, current=10.0, steps=50000))
        ratio = s10 / s2
        assert 1.5 < ratio < 5.0, f"f(10)/f(2) = {ratio:.2f}"
