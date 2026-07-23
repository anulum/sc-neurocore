# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExchangeProtection from former test_ibm_verification_circuits.py

"""Focused suite: TestExchangeProtection from former test_ibm_verification_circuits.py."""

from __future__ import annotations

from tests.ibm_verification_circuits_support import *  # noqa: F403

class TestExchangeProtection:
    def test_weak_exchange_allows_mixing(self):
        p = analytical_singlet_thermal(0.5, HF1, HF2, omega_0=0.5, t=math.pi)
        assert p < 0.75, f"J=0.5 must allow mixing, got {p}"

    def test_strong_exchange_protects(self):
        p = analytical_singlet_thermal(10.0, HF1, HF2, omega_0=0.5, t=math.pi)
        assert p > 0.85, f"J=10 must protect, got {p}"

    def test_zero_hf_preserves(self):
        z = [{"Axx": 0, "Ayy": 0, "Azz": 0, "Axy": 0, "Axz": 0, "Ayz": 0}] * 3
        p = analytical_singlet_thermal(1.0, z, z, 0.0, math.pi, 0.0, 0.0)
        assert p > 0.99
