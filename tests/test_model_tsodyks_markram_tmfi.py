# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestTMFI from former test_model_tsodyks_markram.py

"""Focused suite: TestTMFI from former test_model_tsodyks_markram.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.tsodyks_markram import (
    TsodyksMarkramNeuron,
)
from tests.model_tsodyks_markram_support import (
    _run,
)


class TestTMFI:
    """The current-to-rate response of the TsodyksMarkramNeuron."""

    def test_subthreshold_silent(self) -> None:
        """A subthreshold current gives no spikes."""
        n = TsodyksMarkramNeuron()
        assert len(_run(n, current=10.0, steps=10000)) == 0

    def test_suprathreshold_fires(self) -> None:
        """A suprathreshold current fires repeatedly."""
        n = TsodyksMarkramNeuron()
        assert len(_run(n, current=20.0, steps=10000)) >= 10

    def test_monotonic_fi(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [20.0, 30.0, 50.0]:
            n = TsodyksMarkramNeuron()
            rates.append(len(_run(n, current=I, steps=10000)))
        assert all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
