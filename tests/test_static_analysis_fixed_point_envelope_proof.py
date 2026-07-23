# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFixedPointEnvelopeProof from former test_static_analysis.py

"""Focused suite: TestFixedPointEnvelopeProof from former test_static_analysis.py."""

from __future__ import annotations

from tests.static_analysis_support import *  # noqa: F403

class TestFixedPointEnvelopeProof:
    """Test static Q-format envelope proofs used by precision deployment paths."""

    def test_q1616_safe_mixed_dense_bound_matches_width_contract(self) -> None:
        """A safe mixed-dense envelope proves Q16.16 headroom without cancellation."""
        proof = prove_fixed_point_envelope([531_400], total_bits=32, fractional_bits=16)

        assert isinstance(proof, FixedPointEnvelopeProof)
        assert proof.proof_kind == "signed_symmetric_fixed_point_width"
        assert proof.output_format == "Q16.16"
        assert proof.conservative_safe_bound_code == 2_147_483_647
        assert proof.max_abs_bound_code == 531_400
        assert proof.min_headroom_code == 2_146_952_247
        assert proof.required_total_bits == 21
        assert proof.required_integer_bits == 5
        assert proof.width_headroom_bits == 11
        assert not proof.saturation_required
        assert proof.static_overflow_proven_safe

    def test_q1616_saturating_bound_fails_closed(self) -> None:
        """A bound wider than signed Q16.16 reports saturation instead of safety."""
        proof = prove_fixed_point_envelope(
            [17_454_214_414_336],
            total_bits=32,
            fractional_bits=16,
        )

        assert proof.max_abs_bound_code == 17_454_214_414_336
        assert proof.min_headroom_code == -17_452_066_930_689
        assert proof.required_total_bits == 45
        assert proof.required_integer_bits == 29
        assert proof.width_headroom_bits == -13
        assert proof.saturation_required
        assert not proof.static_overflow_proven_safe

    def test_block_floating_exponent_edge_bound_uses_absolute_vector_max(self) -> None:
        """BFP exponent edge proofs use the largest conservative absolute bound."""
        proof = prove_fixed_point_envelope(
            [1_056_736, -1_069_024],
            total_bits=32,
            fractional_bits=16,
        )

        assert proof.max_abs_bound_code == 1_069_024
        assert proof.required_total_bits == 22
        assert proof.required_integer_bits == 6
        assert proof.width_headroom_bits == 10
        assert not proof.saturation_required
        assert proof.static_overflow_proven_safe

    def test_manifest_is_stable_json_contract(self) -> None:
        """Proof manifests expose the machine-checkable safety gate fields."""
        manifest = prove_fixed_point_envelope([132_850]).manifest()

        assert manifest == {
            "proof_kind": "signed_symmetric_fixed_point_width",
            "output_format": "Q16.16",
            "signed": True,
            "total_bits": 32,
            "fractional_bits": 16,
            "conservative_safe_bound_code": 2_147_483_647,
            "max_abs_bound_code": 132_850,
            "min_headroom_code": 2_147_350_797,
            "required_total_bits": 19,
            "required_integer_bits": 3,
            "width_headroom_bits": 13,
            "saturation_required": False,
            "static_overflow_proven_safe": True,
        }

    def test_unsigned_envelope_rejects_negative_codes(self) -> None:
        """Unsigned Q-code proofs reject signed bounds rather than taking abs()."""
        try:
            prove_fixed_point_envelope([-1], total_bits=16, fractional_bits=8, signed=False)
        except ValueError as exc:
            assert "unsigned fixed-point envelope" in str(exc)
        else:
            raise AssertionError("negative unsigned bound was accepted")

    def test_invalid_format_and_empty_bounds_fail_closed(self) -> None:
        """Malformed proof requests are rejected before producing a manifest."""
        with pytest.raises(ValueError):
            prove_fixed_point_envelope([1], total_bits=0, fractional_bits=0)
        with pytest.raises(ValueError):
            prove_fixed_point_envelope([1], total_bits=16, fractional_bits=16)

        try:
            prove_fixed_point_envelope([])
        except ValueError as exc:
            assert "at least one" in str(exc)
        else:
            raise AssertionError("empty proof bounds were accepted")

    def test_non_integer_bound_codes_are_rejected(self) -> None:
        """Proof inputs must be integer Q-code bounds, not floats or bools."""
        for bad_code in (1.0, True):
            invalid_code: Any = bad_code
            try:
                prove_fixed_point_envelope([invalid_code])
            except TypeError as exc:
                assert "integer Q-code" in str(exc)
            else:
                raise AssertionError(f"non-integer bound accepted: {bad_code!r}")
