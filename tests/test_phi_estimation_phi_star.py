# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPhiStar from former test_phi_estimation.py

"""Focused suite: TestPhiStar from former test_phi_estimation.py."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from sc_neurocore.analysis.phi_estimation import (
    phi_star,
)
from tests.phi_estimation_support import (
    _correlated,
)


class TestPhiStar:
    """The phi-star integration measure."""

    def test_independent_channels_low_phi(self) -> None:
        """Independent channels give low phi."""
        rng = np.random.RandomState(42)
        assert phi_star(rng.randn(4, 200), tau=1, backend="python") < 0.5

    def test_correlated_channels_positive_phi(self) -> None:
        """Correlated channels give positive phi."""
        assert phi_star(_correlated(), tau=1, backend="python") > 0.0

    def test_channel_order_symmetric(self) -> None:
        """Phi does not depend on channel order."""
        rng = np.random.RandomState(42)
        shared = rng.randn(100)
        a = shared + 0.1 * rng.randn(100)
        b = shared + 0.1 * rng.randn(100)
        fwd = phi_star(np.vstack([a, b]), tau=1, backend="python")
        rev = phi_star(np.vstack([b, a]), tau=1, backend="python")
        npt.assert_allclose(fwd, rev, atol=1e-10)

    def test_single_channel_returns_zero(self) -> None:
        """A single channel gives 0."""
        assert phi_star(np.random.randn(1, 100)) == 0.0

    def test_short_data_returns_zero(self) -> None:
        """Too short data gives 0."""
        assert phi_star(np.random.randn(3, 3), tau=2) == 0.0

    def test_nonnegative(self) -> None:
        """Phi is not negative."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            assert phi_star(rng.randn(3, 50), backend="python") >= 0.0

    def test_auto_matches_python_within_tolerance(self) -> None:
        """The automatic backend matches Python."""
        data = _correlated(n_channels=4)
        auto = phi_star(data, tau=1, backend="auto")
        py = phi_star(data, tau=1, backend="python")
        npt.assert_allclose(auto, py, atol=1e-7)

    def test_unknown_backend_rejected(self) -> None:
        """An unknown backend is rejected."""
        with pytest.raises(ValueError, match="not available"):
            phi_star(_correlated(), tau=1, backend="cuda")
