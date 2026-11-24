# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBrainScaleSAdEx from former test_extended_neurons.py

"""Focused suite: TestBrainScaleSAdEx from former test_extended_neurons.py."""

from __future__ import annotations

from sc_neurocore.neurons.models import (
    BrainScaleSAdExNeuron,
)


class TestBrainScaleSAdEx:
    """The BrainScaleS AdEx neuron."""

    def test_fires(self) -> None:
        """The neuron fires under drive."""
        n = BrainScaleSAdExNeuron()
        assert sum(n.step(500.0) for _ in range(2000)) > 0
