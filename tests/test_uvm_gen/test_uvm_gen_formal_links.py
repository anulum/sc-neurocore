# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalLinks from former test_uvm_gen.py

"""Focused suite: TestFormalLinks from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403


class TestFormalLinks:
    def test_generate_links(self):
        gen = UVMGenerator()
        rtl = lif_module()
        links = gen.generate_formal_links(rtl)
        assert len(links) > 0

    def test_link_has_assertion(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        for link in links:
            assert "assert property" in link.assertion_sv

    def test_link_has_cover(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        for link in links:
            assert "cover property" in link.cover_sv

    def test_link_references_reset(self):
        gen = UVMGenerator()
        links = gen.generate_formal_links(lif_module())
        any_rst = any("rst_n" in link.assertion_sv for link in links)
        assert any_rst
