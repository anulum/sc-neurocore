# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFedProx from former test_federated_sc.py

"""Focused suite: TestFedProx from former test_federated_sc.py."""

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    fedprox_gradient,
)


class TestFedProx:
    """The FedProx proximal gradient term."""

    def test_proximal_no_drift(self) -> None:
        """Equal local and global weights leave the gradient unchanged."""
        g = np.array([0.5, 0.3])
        w = np.array([1.0, 1.0])
        result = fedprox_gradient(g, w, w, mu=0.1)
        np.testing.assert_array_almost_equal(result, g)

    def test_proximal_with_drift(self) -> None:
        """Drift adds mu times the weight difference."""
        g = np.array([0.5, 0.3])
        w_local = np.array([2.0, 2.0])
        w_global = np.array([1.0, 1.0])
        result = fedprox_gradient(g, w_local, w_global, mu=0.1)
        expected = g + 0.1 * (w_local - w_global)
        np.testing.assert_array_almost_equal(result, expected)
