# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRecombination from former test_ibm_verification_circuits.py

"""Focused suite: TestRecombination from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403


class TestRecombination:
    def test_recombination_callable(self):
        phi = analytical_singlet_recombination(1.0, HF1, HF2, omega_0=0.5, k_recomb=0.1, n_t=5)
        assert 0.0 < phi < 1.0

    def test_recombination_differs_from_instant(self):
        phi_r = analytical_singlet_recombination(1.0, HF1, HF2, omega_0=0.5, k_recomb=0.1, n_t=10)
        phi_i = analytical_singlet_thermal(1.0, HF1, HF2, omega_0=0.5, t=math.pi)
        assert abs(phi_r - phi_i) > 0.01, "Recomb-weighted must differ from single-t"
