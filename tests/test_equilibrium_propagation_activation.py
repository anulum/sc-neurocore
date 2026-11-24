# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestActivation from former test_equilibrium_propagation.py

"""Focused suite: TestActivation from former test_equilibrium_propagation.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.training.equilibrium_propagation import (
    _rho,
    _rho_prime,
)


class TestActivation:
    """Test hard-sigmoid activation functions."""

    def test_rho_clips_to_01(self) -> None:
        """The activation clips to [0, 1]."""
        x = np.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        result = _rho(x)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.5, 1.0, 1.0])

    def test_rho_prime_in_range(self) -> None:
        """The derivative is 1 inside (0, 1) and 0 outside."""
        x = np.array([-1.0, 0.5, 1.5])
        result = _rho_prime(x)
        np.testing.assert_array_equal(result, [0.0, 1.0, 0.0])

    def test_rho_prime_boundaries(self) -> None:
        """The derivative is 0 at both boundaries."""
        # At exact boundaries, derivative is 0
        assert _rho_prime(np.array([0.0])) == 0.0
        assert _rho_prime(np.array([1.0])) == 0.0
