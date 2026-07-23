# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRadHardLayer from former test_zero_coverage_a.py

"""Focused suite: TestRadHardLayer from former test_zero_coverage_a.py."""

from __future__ import annotations

import sys
from pathlib import Path as _Path

# Ensure same-dir support module is importable under pytest importlib mode.
sys.path.insert(0, str(_Path(__file__).resolve().parent))

from zero_coverage_a_support import *  # noqa: F403

class TestRadHardLayer:
    def test_forward(self):
        from exotic.space import RadHardLayer

        r = RadHardLayer(n_inputs=3, n_neurons=2, length=64)
        result = r.forward([0.5, 0.3, 0.7])
        assert isinstance(result, np.ndarray)
        assert result.shape == (2,)
