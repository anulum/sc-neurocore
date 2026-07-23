# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDriftAutoCorrector from former test_twinsync.py

"""Focused suite: TestDriftAutoCorrector from former test_twinsync.py."""

from __future__ import annotations

from twinsync_support import *  # noqa: F403

class TestDriftAutoCorrector:
    def test_no_correction_within_tolerance(self):
        dac = DriftAutoCorrector(max_drift_ns=5000)
        assert dac.check_and_correct(1000, 999) is None

    def test_correction_on_large_drift(self):
        dac = DriftAutoCorrector(max_drift_ns=5000)
        dc = dac.check_and_correct(100000, 1000)
        assert dc is not None
        assert dc.correction_ns > 0
        assert dac.total_corrections == 1
