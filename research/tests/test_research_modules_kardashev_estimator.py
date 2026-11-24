# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestKardashevEstimator from former test_research_modules.py

"""Focused suite: TestKardashevEstimator from former test_research_modules.py."""

from __future__ import annotations

from speculative.analysis_kardashev import KardashevEstimator
import pytest


class TestKardashevEstimator:
    """The Kardashev-scale estimate of a power budget."""

    def test_zero_power(self) -> None:
        """Zero power gives type 0."""
        assert KardashevEstimator.calculate_type(0) == 0.0

    def test_negative_power(self) -> None:
        """Negative power gives type 0."""
        assert KardashevEstimator.calculate_type(-100) == 0.0

    def test_type_1_civilization(self) -> None:
        """The type I power gives type 1."""
        # Type 1 = 10^16 W
        k = KardashevEstimator.calculate_type(1e16)
        assert k == pytest.approx(1.0)

    def test_type_2_civilization(self) -> None:
        """The type II power gives type 2."""
        # Type 2 = 10^26 W
        k = KardashevEstimator.calculate_type(1e26)
        assert k == pytest.approx(2.0)

    def test_estimate_from_compute(self) -> None:
        """The compute-based estimate gives type 1 at the type I power."""
        k = KardashevEstimator.estimate_from_compute(1e37, efficiency_j_per_op=1e-21)
        assert k == pytest.approx(1.0)
