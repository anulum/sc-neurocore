# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPQC from former test_intelligence_security_and_compliance.py

"""Focused suite: TestPQC from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

import unittest
from sc_neurocore.compiler.intelligence import (
    protect_ip_pqc,
)


class TestPQC(unittest.TestCase):
    """Post-quantum signing of bitstreams."""

    def test_basic(self) -> None:
        """Dilithium gives a quantum-safe signature."""
        r = protect_ip_pqc("sc_lif", {"v": "a"})
        self.assertTrue(r.quantum_safe)
        self.assertEqual(r.algorithm, "CRYSTALS-Dilithium")
        self.assertEqual(len(r.signature_hex), 32)
        self.assertEqual(r.key_size_bits, 1952)

    def test_security_levels(self) -> None:
        """A higher security level uses a larger key."""
        r2 = protect_ip_pqc("m", {"v": "a"}, security_level=2)
        r5 = protect_ip_pqc("m", {"v": "a"}, security_level=5)
        self.assertLess(r2.key_size_bits, r5.key_size_bits)

    def test_deterministic(self) -> None:
        """The same input gives the same signature."""
        r1 = protect_ip_pqc("m", {"v": "a"})
        r2 = protect_ip_pqc("m", {"v": "a"})
        self.assertEqual(r1.signature_hex, r2.signature_hex)
