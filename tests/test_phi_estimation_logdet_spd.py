# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestLogdetSpd from former test_phi_estimation.py

"""Focused suite: TestLogdetSpd from former test_phi_estimation.py."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
from sc_neurocore.analysis.phi_estimation import (
    _logdet_spd,
)


class TestLogdetSpd:
    """Log-determinants of SPD matrices."""

    def test_diagonal(self) -> None:
        """A diagonal matrix gives the log of its product."""
        m = np.diag([2.0, 8.0])
        npt.assert_allclose(_logdet_spd(m), np.log(16.0))

    def test_matches_slogdet(self) -> None:
        """The result matches NumPy's slogdet."""
        rng = np.random.RandomState(0)
        a = rng.randn(5, 5)
        spd = a @ a.T + np.eye(5)
        _, ref = np.linalg.slogdet(spd)
        npt.assert_allclose(_logdet_spd(spd), ref, atol=1e-10)
