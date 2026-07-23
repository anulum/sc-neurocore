# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCTCLayer from former test_zero_coverage_a.py

"""Focused suite: TestCTCLayer from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestCTCLayer:
    def test_self_consistency(self):
        from meta.time_travel import CTCLayer

        t = CTCLayer(n_bits=8, max_iterations=50)
        result = t.compute_self_consistency(
            lambda x: np.bitwise_xor(x, np.ones_like(x, dtype=np.uint8))
        )
        assert isinstance(result, np.ndarray)
