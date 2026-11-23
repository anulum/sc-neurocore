# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestAssertionBind from former test_uvm_gen.py

"""Focused suite: TestAssertionBind from former test_uvm_gen.py."""

from __future__ import annotations

from sc_neurocore.uvm_gen.uvm_gen import (
    UVMGenerator,
)
from uvm_gen_support import (
    lif_module,
)


class TestAssertionBind:
    """The generated SystemVerilog assertion bind file."""

    def test_bind_generated(self) -> None:
        """A bind file is generated."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.bind_sv) > 0

    def test_bind_has_assertions(self) -> None:
        """The bind file holds assertion properties."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "assert property" in bench.bind_sv

    def test_bind_has_cover(self) -> None:
        """The bind file holds cover properties."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "cover property" in bench.bind_sv

    def test_bind_has_reset_check(self) -> None:
        """The bind file checks the reset signal."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "rst_n" in bench.bind_sv

    def test_bind_module_name(self) -> None:
        """The bind file names its assertion module after the DUT."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "sc_lif_neuron_assertions" in bench.bind_sv

    def test_bind_in_dict(self) -> None:
        """The bind file appears in the file dictionary."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "sc_lif_neuron_bind.sv" in d
