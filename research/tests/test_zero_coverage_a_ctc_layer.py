# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestCTCLayer from former test_zero_coverage_a.py

"""Focused suite: TestCTCLayer from former test_zero_coverage_a.py."""

from __future__ import annotations

from typing import Any, cast


import numpy as np


class TestCTCLayer:
    """The closed-timelike-curve layer research module."""

    def test_self_consistency(self) -> None:
        """The self-consistency solve returns an array (the check is on the type only)."""
        from meta.time_travel import CTCLayer

        t = CTCLayer(n_bits=8, max_iterations=50)
        result = cast(Any, t).compute_self_consistency(
            lambda x: np.bitwise_xor(x, np.ones_like(x, dtype=np.uint8))
        )
        assert isinstance(result, np.ndarray)
