# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMartinottiAdaptation from former test_model_martinotti_neuron.py

"""Focused suite: TestMartinottiAdaptation from former test_model_martinotti_neuron.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.martinotti_neuron import (
    MartinottiNeuron,
)
from tests.model_martinotti_neuron_support import (
    _spikes,
)


class TestMartinottiAdaptation:
    """The M-current adaptation of the MartinottiNeuron."""

    def test_m_current_block_changes_firing(self) -> None:
        """M current block changes firing."""
        intact = _spikes(MartinottiNeuron(), 5.0, 40000)
        blocked = _spikes(MartinottiNeuron(g_m=0.0), 5.0, 40000)
        assert intact != blocked
