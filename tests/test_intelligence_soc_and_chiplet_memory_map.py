# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestMemoryMap from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestMemoryMap from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations


class TestMemoryMap:
    """Generating a memory map."""

    def test_basic(self) -> None:
        """The map has entries and an address decoder."""
        from sc_neurocore.compiler.intelligence import generate_memory_map

        m = generate_memory_map("sc_lif", {"v": "a", "u": "b"})
        assert m.total_bytes > 0
        assert "addr_dec" in m.decoder_verilog
        assert len(m.entries) > 0

    def test_base_address(self) -> None:
        """A custom base address is used."""
        from sc_neurocore.compiler.intelligence import generate_memory_map

        m = generate_memory_map("sc_lif", {"v": "a"}, base_address=0x2000)
        assert m.base_address == 0x2000
