# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIPObfuscation from former test_intelligence_security_and_compliance.py

"""Focused suite: TestIPObfuscation from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403


class TestIPObfuscation:
    def test_basic(self):
        from sc_neurocore.compiler.intelligence import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"})
        assert r.key_bits == 64
        assert "logic_locking" in r.techniques_applied
        assert r.obfuscated_signals > r.original_signals

    def test_custom_key(self):
        from sc_neurocore.compiler.intelligence import obfuscate_ip

        r = obfuscate_ip("sc_lif", {"v": "a + b"}, key_length=128)
        assert r.key_bits == 128
