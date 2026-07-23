# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSSTBetaMRegression from former test_model_sst_neuron.py

"""Focused suite: TestSSTBetaMRegression from former test_model_sst_neuron.py."""

from __future__ import annotations

from tests.model_sst_neuron_support import *  # noqa: F403

class TestSSTBetaMRegression:
    """Guards the corrected β_m offset against the depolarisation-block bug."""

    def test_firing_rate_increases_with_current(self):
        low = _spikes(SSTNeuron(), 0.5, 30000)
        mid = _spikes(SSTNeuron(), 2.0, 30000)
        high = _spikes(SSTNeuron(), 5.0, 30000)
        assert low < mid < high

    def test_no_depolarisation_block_at_strong_drive(self):
        # The bug stuck V near threshold and capped firing at three spikes for any
        # stimulus; healthy kinetics fire repetitively under strong drive.
        assert _spikes(SSTNeuron(), 5.0, 40000) > 100

    def test_membrane_recovers_below_threshold_after_drive(self):
        n = SSTNeuron()
        for _ in range(40000):
            n.step(2.0)
        assert n.v < n.v_threshold
