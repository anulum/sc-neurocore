# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRecursiveSelfImprover from former test_zero_coverage_a.py

"""Focused suite: TestRecursiveSelfImprover from former test_zero_coverage_a.py."""

from __future__ import annotations

from typing import Any, cast


import numpy as np


class TestRecursiveSelfImprover:
    """The recursive self-improver research module."""

    def test_improve(self) -> None:
        """An improvement step returns a float (the check is on the type only)."""
        from meta.singularity import RecursiveSelfImprover
        from sc_neurocore.layers.vectorized_layer import VectorizedSCLayer

        s = RecursiveSelfImprover()
        layer = VectorizedSCLayer(n_inputs=4, n_neurons=2, length=32)
        result = cast(Any, s).improve(layer)
        assert isinstance(result, (float, np.floating))
