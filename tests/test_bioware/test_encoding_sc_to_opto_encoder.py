# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSCToOptoEncoder from former test_encoding.py

"""Focused suite: TestSCToOptoEncoder from former test_encoding.py."""

from __future__ import annotations

from tests.test_bioware.encoding_support import *  # noqa: F403

class TestSCToOptoEncoder:
    def test_encode(self) -> None:
        bs = {0: np.ones(128, dtype=np.uint8), 1: np.zeros(128, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=5.0)
        pulses = enc.encode(bs)
        assert len(pulses) == 1  # neuron 1 is silent, skipped

    def test_intensity_scaling(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(max_intensity_mw_mm2=10.0)
        pulses = enc.encode(bs)
        assert pulses[0].intensity_mw_mm2 == 10.0

    def test_wavelength(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(wavelength_nm=590)
        pulses = enc.encode(bs)
        assert pulses[0].wavelength_nm == 590

    def test_duration_range(self) -> None:
        bs = {0: np.ones(100, dtype=np.uint8)}
        enc = SCToOptoEncoder(min_pulse_ms=1.0, max_pulse_ms=50.0)
        pulses = enc.encode(bs)
        assert enc.min_pulse_ms <= pulses[0].duration_ms <= enc.max_pulse_ms
