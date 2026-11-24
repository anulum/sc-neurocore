# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestSurrogateGradient from former test_advanced_plasticity.py

"""Focused suite: TestSurrogateGradient from former test_advanced_plasticity.py."""

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.learning.advanced import (
    _fast_sigmoid_surrogate,
)


class TestSurrogateGradient:
    """The fast-sigmoid surrogate gradient."""

    def test_peak_at_threshold(self) -> None:
        """The surrogate peaks at the threshold."""
        grad = _fast_sigmoid_surrogate(np.array([1.0]))
        assert grad[0] > 0
        assert grad[0] == pytest.approx(25.0, rel=0.01)

    def test_decays_away(self) -> None:
        """The surrogate falls away from the threshold."""
        at_thresh = _fast_sigmoid_surrogate(np.array([1.0]))[0]
        away = _fast_sigmoid_surrogate(np.array([2.0]))[0]
        assert away < at_thresh
