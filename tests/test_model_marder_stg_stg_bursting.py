# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSTGBursting from former test_model_marder_stg.py

"""Focused suite: TestSTGBursting from former test_model_marder_stg.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.neurons.models.marder_stg import (
    MarderSTGNeuron,
)
from tests.model_marder_stg_support import (
    _run,
)


class TestSTGBursting:
    """Bursting, calcium accumulation and voltage bounds of the MarderSTGNeuron."""

    def test_fires_at_zero_current(self) -> None:
        """The model fires at zero current."""
        assert len(_run(MarderSTGNeuron(), current=0.0, steps=50_000)) >= 10

    def test_bursting_pattern(self) -> None:
        """Without drive, spikes come in bursts separated by long gaps."""
        spikes = _run(MarderSTGNeuron(), current=0.0, steps=100_000)
        isis = np.diff(spikes)
        assert isis.max() > 3 * np.median(isis), "expected bursts separated by quiescent gaps"

    def test_calcium_accumulates_during_spiking(self) -> None:
        """Calcium accumulates during spiking."""
        n = MarderSTGNeuron()
        for _ in range(50_000):
            n.step(0.0)
        assert n.ca > 1.0

    def test_calcium_non_negative(self) -> None:
        """Calcium non-negative."""
        n = MarderSTGNeuron()
        for _ in range(100_000):
            n.step(2.0)
            assert n.ca >= 0.0

    def test_voltage_bounded(self) -> None:
        """The voltage stays inside physiological bounds."""
        n = MarderSTGNeuron()
        vs = [n.v for _ in range(50_000) if (n.step(0.0) or True)]
        assert min(vs) > -100.0 and max(vs) < 80.0
