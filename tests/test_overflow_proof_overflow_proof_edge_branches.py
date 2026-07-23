# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestOverflowProofEdgeBranches from former test_overflow_proof.py

"""Focused suite: TestOverflowProofEdgeBranches from former test_overflow_proof.py."""

from __future__ import annotations

from tests.overflow_proof_support import *  # noqa: F403

class TestOverflowProofEdgeBranches:
    def test_unary_plus_preserves_operand_interval(self) -> None:
        result = prove_no_overflow("+x", {"x": (1.0, 2.0)})
        assert result.proven_safe is True

    def test_unbound_variable_raises_keyerror(self) -> None:
        import pytest

        with pytest.raises(KeyError, match="No bounds for variable 'y'"):
            prove_no_overflow("y", {"x": (1.0, 2.0)})

    def test_non_numeric_constant_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Unsupported constant"):
            prove_no_overflow("'abc'", {})

    def test_numeric_constant_is_a_point_interval(self) -> None:
        # A literal in the expression is evaluated as a degenerate point interval.
        result = prove_no_overflow("x + 5", {"x": (1.0, 2.0)})
        assert result.proven_safe is True

    def test_required_total_bits_for_bound_edges(self) -> None:
        from sc_neurocore.compiler.overflow_proof import _required_total_bits_for_bound

        assert _required_total_bits_for_bound(0, signed=True) == 1
        assert _required_total_bits_for_bound(255, signed=False) == 8

    def test_envelope_rejects_negative_fractional_bits(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="fractional_bits must be non-negative"):
            prove_fixed_point_envelope([100], fractional_bits=-1)

    def test_envelope_supports_unsigned_format(self) -> None:
        proof = prove_fixed_point_envelope([100], signed=False)
        assert proof.proof_kind == "unsigned_fixed_point_width"
