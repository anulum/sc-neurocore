# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAssertionBind from former test_uvm_gen.py

"""Focused suite: TestAssertionBind from former test_uvm_gen.py."""

from __future__ import annotations

from uvm_gen_support import *  # noqa: F403

class TestAssertionBind:
    def test_bind_generated(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.bind_sv) > 0

    def test_bind_has_assertions(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "assert property" in bench.bind_sv

    def test_bind_has_cover(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "cover property" in bench.bind_sv

    def test_bind_has_reset_check(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "rst_n" in bench.bind_sv

    def test_bind_module_name(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "sc_lif_neuron_assertions" in bench.bind_sv

    def test_bind_in_dict(self):
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "sc_lif_neuron_bind.sv" in d
