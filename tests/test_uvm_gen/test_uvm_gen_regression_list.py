# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestRegressionList from former test_uvm_gen.py

"""Focused suite: TestRegressionList from former test_uvm_gen.py."""

from __future__ import annotations

from sc_neurocore.uvm_gen.uvm_gen import (
    UVMGenerator,
)
from uvm_gen_support import (
    lif_module,
)


class TestRegressionList:
    """The generated regression list."""

    def test_regression_list_generated(self) -> None:
        """A regression list is generated."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert len(bench.regression_list) > 0

    def test_regression_has_tests(self) -> None:
        """The list holds random, corner and LFSR tests."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        assert "random" in bench.regression_list
        assert "corner" in bench.regression_list
        assert "lfsr" in bench.regression_list

    def test_regression_in_dict(self) -> None:
        """The regression list appears in the file dictionary."""
        gen = UVMGenerator()
        bench = gen.generate(lif_module())
        d = bench.to_dict()
        assert "regression.list" in d
