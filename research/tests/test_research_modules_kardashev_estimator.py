# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKardashevEstimator from former test_research_modules.py

"""Focused suite: TestKardashevEstimator from former test_research_modules.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from research_modules_support import *  # noqa: F403

class TestKardashevEstimator:
    def test_zero_power(self):
        assert KardashevEstimator.calculate_type(0) == 0.0

    def test_negative_power(self):
        assert KardashevEstimator.calculate_type(-100) == 0.0

    def test_type_1_civilization(self):
        # Type 1 = 10^16 W
        k = KardashevEstimator.calculate_type(1e16)
        assert k == pytest.approx(1.0)

    def test_type_2_civilization(self):
        # Type 2 = 10^26 W
        k = KardashevEstimator.calculate_type(1e26)
        assert k == pytest.approx(2.0)

    def test_estimate_from_compute(self):
        k = KardashevEstimator.estimate_from_compute(1e37, efficiency_j_per_op=1e-21)
        assert k == pytest.approx(1.0)
