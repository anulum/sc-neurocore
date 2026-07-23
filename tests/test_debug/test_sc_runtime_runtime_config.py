# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRuntimeConfig from former test_sc_runtime.py

"""Focused suite: TestRuntimeConfig from former test_sc_runtime.py."""

from __future__ import annotations

from sc_runtime_support import *  # noqa: F403

class TestRuntimeConfig:
    def test_default_values(self):
        c = RuntimeConfig()
        assert c.bitstream_length == 256
        assert c.decorrelator == DecorrelatorType.LFSR
        assert not c.ecc_enabled

    def test_effective_length_no_ecc(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=False)
        assert c.effective_length == 256

    def test_effective_length_with_ecc(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.HAMMING)
        assert c.effective_length == 256 + (256 // 4) * 3

    def test_effective_length_secded(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.SECDED)
        assert c.effective_length == 256 + (256 // 4) * 4

    def test_effective_length_parity(self):
        c = RuntimeConfig(bitstream_length=256, ecc_enabled=True, ecc_mode=ECCMode.PARITY)
        assert c.effective_length == 256 + (256 // 8)

    def test_copy_independent(self):
        c = RuntimeConfig(bitstream_length=512)
        d = c.copy()
        d.bitstream_length = 1024
        assert c.bitstream_length == 512

    def test_copy_preserves_ecc_mode(self):
        c = RuntimeConfig(ecc_mode=ECCMode.SECDED)
        d = c.copy()
        assert d.ecc_mode == ECCMode.SECDED
