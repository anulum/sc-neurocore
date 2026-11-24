# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNoSpikeModel from former test_neuron_families.py

"""Focused suite: TestNoSpikeModel from former test_neuron_families.py."""

from __future__ import annotations

from tests.neuron_families_support import (
    _make_leaky_no_spike,
)


class TestNoSpikeModel:
    """A leaky model with no input that must never spike."""

    def test_subthreshold_never_spikes(self) -> None:
        """Without input the leaky model never spikes in 10000 steps."""
        neuron = _make_leaky_no_spike()
        for _ in range(10000):
            assert not neuron.step(I=0.0), "subthreshold model spiked"

    def test_decays_to_zero(self) -> None:
        """Without input the voltage decays to near zero."""
        neuron = _make_leaky_no_spike()
        for _ in range(1000):
            neuron.step(I=0.0)
        assert abs(neuron.state["v"]) < 0.001
