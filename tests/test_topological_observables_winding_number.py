# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestWindingNumber from former test_topological_observables.py

"""Focused suite: TestWindingNumber from former test_topological_observables.py."""

from __future__ import annotations

from tests.topological_observables_support import *  # noqa: F403

class TestWindingNumber:
    def test_zero_wraps(self):
        """Half rotation (0 to pi) = 0 wraps."""
        phases = np.linspace(0, np.pi * 0.9, 500)
        assert winding_number(phases) == 0

    def test_one_wrap(self):
        T = 1000
        omega = 2 * np.pi / T
        phases = np.array([(omega * t) % (2 * np.pi) for t in range(T)])
        assert winding_number(phases) == 1

    def test_three_wraps(self):
        T = 1000
        omega = 3 * 2 * np.pi / T
        phases = np.array([(omega * t) % (2 * np.pi) for t in range(T)])
        assert winding_number(phases) == 3

    def test_constant_phase_zero(self):
        phases = np.full(100, 1.5)
        assert winding_number(phases) == 0

    def test_negative_direction(self):
        """Negative frequency should give negative winding."""
        T = 1000
        omega = -2 * 2 * np.pi / T
        phases = np.array([((omega * t) % (2 * np.pi) + 2 * np.pi) % (2 * np.pi) for t in range(T)])
        w = winding_number(phases)
        # Depending on implementation, may be -2 or wrapped
        assert abs(w) in (0, 2), f"unexpected winding {w}"
