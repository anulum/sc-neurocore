# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEquivChecker from former test_nas.py

"""Focused suite: TestEquivChecker from former test_nas.py."""

from __future__ import annotations

from sc_neurocore.nas.equiv import (
    check_equivalence,
    generate_miter,
    generate_sby,
)


class TestEquivChecker:
    """Formal equivalence checking."""

    def test_check_no_run(self) -> None:
        """A check without running passes and says so."""
        r = check_equivalence(run=False)
        assert r.passed is True
        assert "not run" in r.log

    def test_generate_miter(self) -> None:
        """The miter instantiates both designs."""
        v = generate_miter("sc_lif_neuron", "sc_lif_reference", "equiv_test")
        assert "module equiv_test" in v
        assert "sc_lif_neuron" in v
        assert "sc_lif_reference" in v
        assert "assert" in v

    def test_generate_sby(self) -> None:
        """The SBY file runs to depth 20 on the miter."""
        s = generate_sby("equiv_test", ["a.v", "b.v", "c.v"], depth=20)
        assert "depth 20" in s
        assert "read -formal a.v" in s
        assert "prep -top equiv_test" in s

    def test_generate_miter_custom_width(self) -> None:
        """A custom data width reaches the miter."""
        v = generate_miter("dut", "ref", "top", data_width=8, fraction=4)
        assert "DATA_WIDTH = 8" in v
        assert "FRACTION = 4" in v
