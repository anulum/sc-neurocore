# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLoihiCUBA from former test_extended_neurons.py

"""Focused suite: TestLoihiCUBA from former test_extended_neurons.py."""

from __future__ import annotations

from sc_neurocore.neurons.models import (
    LoihiCUBANeuron,
)


class TestLoihiCUBA:
    """The Loihi CUBA neuron."""

    def test_fires(self) -> None:
        """The neuron fires under drive."""
        n = LoihiCUBANeuron()
        assert sum(n.step(200) for _ in range(100)) > 0
