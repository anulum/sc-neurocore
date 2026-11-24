# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestYamadaFI from former test_model_yamada.py

"""Focused suite: TestYamadaFI from former test_model_yamada.py."""

from __future__ import annotations

from sc_neurocore.neurons.models.yamada import (
    YamadaNeuron,
)
from tests.model_yamada_support import (
    _run,
)


class TestYamadaFI:
    """The current-to-rate response of the YamadaNeuron."""

    def test_silent_at_zero(self) -> None:
        """Zero current gives no spikes."""
        n = YamadaNeuron()
        assert len(_run(n, current=0.0, steps=50000)) == 0

    def test_fires_at_high_current(self) -> None:
        """The model fires at high current."""
        n = YamadaNeuron()
        assert len(_run(n, current=50.0, steps=200000)) >= 10

    def test_rate_increases_with_current(self) -> None:
        """The firing rate rises with the current."""
        n1 = YamadaNeuron()
        n2 = YamadaNeuron()
        s1 = len(_run(n1, current=30.0, steps=200000))
        s2 = len(_run(n2, current=100.0, steps=200000))
        assert s2 > s1
