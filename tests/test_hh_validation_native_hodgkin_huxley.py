# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestNativeHodgkinHuxley from former test_hh_validation.py

"""Focused suite: TestNativeHodgkinHuxley from former test_hh_validation.py."""

from __future__ import annotations

from tests.hh_validation_support import *  # noqa: F403

class TestNativeHodgkinHuxley:
    """Cross-validate the sc_neurocore HodgkinHuxleyNeuron."""

    def test_native_fires(self):
        """Native HH produces spikes at I=10."""
        hh = HodgkinHuxleyNeuron()
        spikes = sum(hh.step(10.0) for _ in range(50))
        assert spikes > 0, "0 spikes in 50ms"

    def test_native_resting(self):
        """Native HH settles near -65 mV with no current."""
        hh = HodgkinHuxleyNeuron()
        for _ in range(100):
            hh.step(0.0)
        assert -70 < hh.v < -60, f"V={hh.v:.2f}"
