# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRefractoryPeriod from former test_acquisition.py

"""Focused suite: TestRefractoryPeriod from former test_acquisition.py."""

from __future__ import annotations

from sc_neurocore.bioware.bioware import (
    MEAConfig,
    SpikeDetector,
)
from tests.test_bioware.acquisition_support import (
    _synth_voltage,
)


class TestRefractoryPeriod:
    """The spike detector's refractory period."""

    def test_refractory_reduces_spikes(self) -> None:
        """A refractory period never adds spikes."""
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        det_no_ref = SpikeDetector(config=cfg, refractory_samples=0)
        det_with_ref = SpikeDetector(config=cfg, refractory_samples=50)
        data = _synth_voltage()
        spikes_no = det_no_ref.detect(data)
        spikes_yes = det_with_ref.detect(data)
        assert len(spikes_yes) <= len(spikes_no)

    def test_refractory_zero_matches_original(self) -> None:
        """With no refractory period spikes are still found."""
        cfg = MEAConfig(num_channels=10, spike_threshold_sigma=3.0)
        # refractory_samples=0 means no refractory filter
        det = SpikeDetector(config=cfg, refractory_samples=0)
        data = _synth_voltage()
        spikes = det.detect(data)
        assert len(spikes) > 0
