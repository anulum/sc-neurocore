# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestGFSDynamics from former test_model_golomb_fs.py

"""Focused suite: TestGFSDynamics from former test_model_golomb_fs.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.neurons.models.golomb_fs import (
    GolombFSNeuron,
)
from tests.model_golomb_fs_support import (
    _run,
)


class TestGFSDynamics:
    """Dynamics of the GolombFSNeuron."""

    def test_fires_under_drive(self) -> None:
        """The model fires under drive."""
        n = GolombFSNeuron()
        spikes = _run(n, current=5.0, steps=5000)
        assert len(spikes) >= 10

    def test_subthreshold_silent(self) -> None:
        """A subthreshold current gives no spikes."""
        n = GolombFSNeuron()
        assert len(_run(n, current=0.5, steps=2000)) == 0

    def test_high_sustained_rate(self) -> None:
        """FS interneurons sustain high firing rates."""
        n = GolombFSNeuron()
        spikes = _run(n, current=10.0, steps=5000)
        assert len(spikes) >= 20

    def test_rate_monotonic(self) -> None:
        """The firing rate does not fall as the current rises."""
        rates = []
        for I in [2.0, 5.0, 10.0]:
            n = GolombFSNeuron()
            rates.append(len(_run(n, current=I, steps=5000)))
        assert rates[-1] >= rates[0]

    @pytest.mark.parametrize("current", [0.0, 2.0, 5.0, 10.0, 20.0])
    def test_fi_sweep(self, current: float) -> None:
        """The state stays finite across a current sweep."""
        n = GolombFSNeuron()
        for _ in range(2000):
            n.step(current)
        assert np.isfinite(n.v)

    def test_voltage_bounded(self) -> None:
        """The voltage stays inside physiological bounds."""
        n = GolombFSNeuron()
        vs = []
        for _ in range(2000):
            n.step(5.0)
            vs.append(n.v)
        assert min(vs) > -100 and max(vs) < 60
