# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPrecisionTracker from former test_wave4.py

"""Focused suite: TestPrecisionTracker from former test_wave4.py."""

from __future__ import annotations

from wave4_support import *  # noqa: F403

class TestPrecisionTracker:
    def test_ema(self):
        pt = PrecisionTracker(alpha=0.5)
        pt.update(1.0)
        assert pt.ema == 1.0
        pt.update(0.0)
        assert pt.ema == pytest.approx(0.5)
