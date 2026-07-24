# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormalEquivalence from former test_intelligence_verification_and_safety.py

"""Focused suite: TestFormalEquivalence from former test_intelligence_verification_and_safety.py."""

from __future__ import annotations

from tests.intelligence_verification_and_safety_support import *  # noqa: F403


class TestFormalEquivalence:
    """Formal equivalence proof skeleton."""

    def test_basic_sketch(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch(
            "sc_lif",
            {"v": "a + b * c"},
        )
        assert s.module_name == "sc_lif"
        assert len(s.proof_steps) >= 5
        assert len(s.assertions) == 1
        assert s.quantisation_bound > 0

    def test_multi_equation(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch(
            "sc_izh",
            {"v": "a * b + c", "u": "d * e"},
        )
        assert len(s.assertions) == 2
        assert "CONCLUSION" in s.proof_steps[-1]

    def test_sva_format(self):
        from sc_neurocore.compiler.intelligence import (
            generate_equivalence_sketch,
        )

        s = generate_equivalence_sketch("sc_lif", {"v": "a + b"})
        assert "assert property" in s.assertions[0]
        assert "posedge clk" in s.assertions[0]
