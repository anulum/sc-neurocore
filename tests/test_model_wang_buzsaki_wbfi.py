# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWBFI from former test_model_wang_buzsaki.py

"""Focused suite: TestWBFI from former test_model_wang_buzsaki.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.wang_buzsaki import (
    WangBuzsakiNeuron,
)
from tests.model_wang_buzsaki_support import (
    _run,
)


class TestWBFI:
    """The current-to-rate response of the WangBuzsakiNeuron."""

    def test_subthreshold_silent(self) -> None:
        """A subthreshold current gives no spikes."""
        n = WangBuzsakiNeuron()
        assert len(_run(n, current=0.0, steps=20000)) == 0

    def test_monotonic_fi(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [0.5, 1.0, 2.0, 5.0]:
            n = WangBuzsakiNeuron()
            rates.append(len(_run(n, current=I, steps=20000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))

    def test_fast_spiking_at_high_current(self) -> None:
        """At I=10, frequency >> gamma band (fast-spiking characteristic)."""
        n = WangBuzsakiNeuron()
        spikes = _run(n, current=10.0, steps=20000)
        assert len(spikes) >= 1000  # very high rate
