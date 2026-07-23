# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestBitstreamEncryption from former test_intelligence_security_and_compliance.py

"""Focused suite: TestBitstreamEncryption from former test_intelligence_security_and_compliance.py."""

from __future__ import annotations

from tests.intelligence_security_and_compliance_support import *  # noqa: F403

class TestBitstreamEncryption:
    """AES-256 bitstream encryption for secure boot."""

    def test_xilinx_encryption(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_lif", vendor="xilinx")
        assert "BITSTREAM.ENCRYPTION.ENCRYPT YES" in tcl
        assert "sc_lif.nky" in tcl
        assert "SECURITY_LEVEL" in tcl

    def test_intel_encryption(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("sc_hh", vendor="intel")
        assert "ENCRYPTION_KEY_SOURCE" in tcl
        assert "ENABLE_CONFIGURATION_BITSTREAM_ENCRYPTION ON" in tcl
        assert "ANTI_TAMPER" in tcl

    def test_key_source_efuse(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="efuse",
        )
        assert "EFUSE" in tcl

    def test_key_source_bbram(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption(
            "sc_lif",
            key_source="bbram",
        )
        assert "BBRAM" in tcl

    def test_module_name_in_output(self):
        from sc_neurocore.compiler.intelligence import (
            generate_bitstream_encryption,
        )

        tcl = generate_bitstream_encryption("my_neuron_design")
        assert "my_neuron_design" in tcl
