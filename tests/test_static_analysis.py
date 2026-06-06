# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for static analysis (guard bits, overflow proof, SVA)

"""Tests for guard-bit computation, formal overflow proofs, and SVA generation."""

from __future__ import annotations


from sc_neurocore.compiler.static_analysis import (
    FixedPointEnvelopeProof,
    Interval,
    compute_guard_bits,
    compute_guard_bits_multi,
    generate_sva,
    prove_fixed_point_envelope,
    prove_no_overflow,
)


# ═══════════════════════════════════════════════════════════════════════
# Guard-Bit Tests
# ═══════════════════════════════════════════════════════════════════════


class TestGuardBits:
    """Test guard-bit auto-computation from AST analysis."""

    def test_no_additions(self) -> None:
        """Expression with only multiplication needs 0 guard bits."""
        assert compute_guard_bits("a * b") == 0

    def test_single_addition(self) -> None:
        """One addition needs 1 guard bit."""
        assert compute_guard_bits("a + b") == 1

    def test_single_subtraction(self) -> None:
        """Subtraction also counts as an addition."""
        assert compute_guard_bits("a - b") == 1

    def test_three_additions(self) -> None:
        """Three additions (4 terms) needs 2 guard bits."""
        assert compute_guard_bits("a + b + c + d") >= 2

    def test_complex_ode(self) -> None:
        """LIF ODE: -(v - v_rest) / tau_m + R * I / C has additions."""
        bits = compute_guard_bits("-(v - v_rest) / tau_m + R * I / C")
        assert bits >= 1

    def test_multi_variable(self) -> None:
        """Multi-ODE system returns per-variable guard bits."""
        eqs = {
            "v": "-(v - v_rest) / tau + I",
            "u": "a * (b * v - u)",
        }
        result = compute_guard_bits_multi(eqs)
        assert "v" in result
        assert "u" in result
        assert result["v"] >= 1
        assert result["u"] >= 1

    def test_constant_expression(self) -> None:
        """A constant has 0 additions."""
        assert compute_guard_bits("42") == 0

    def test_nested_multiply(self) -> None:
        """Nested multiplies with no additions need 0 guard bits."""
        assert compute_guard_bits("a * b * c") == 0


# ═══════════════════════════════════════════════════════════════════════
# Interval Arithmetic Tests
# ═══════════════════════════════════════════════════════════════════════


class TestIntervalArithmetic:
    """Test the Interval class used in overflow proofs."""

    def test_addition(self) -> None:
        """[1,2] + [3,4] = [4,6]."""
        r = Interval(1, 2) + Interval(3, 4)
        assert r.lo == 4 and r.hi == 6

    def test_subtraction(self) -> None:
        """[1,5] - [2,3] = [-2,3]."""
        r = Interval(1, 5) - Interval(2, 3)
        assert r.lo == -2 and r.hi == 3

    def test_multiplication(self) -> None:
        """[-2,3] * [1,4] = [-8,12]."""
        r = Interval(-2, 3) * Interval(1, 4)
        assert r.lo == -8 and r.hi == 12

    def test_division(self) -> None:
        """[6,12] / [2,3] = [2,6]."""
        r = Interval(6, 12) / Interval(2, 3)
        assert r.lo == 2.0 and r.hi == 6.0

    def test_division_by_zero(self) -> None:
        """Division by interval containing zero returns (-inf, inf)."""
        r = Interval(1, 2) / Interval(-1, 1)
        assert r.lo == float("-inf")

    def test_negation(self) -> None:
        """-[2,5] = [-5,-2]."""
        r = -Interval(2, 5)
        assert r.lo == -5 and r.hi == -2

    def test_contains(self) -> None:
        """[3,7] is contained in [-128, 127]."""
        assert Interval(3, 7).contains(-128, 127)
        assert not Interval(-200, 7).contains(-128, 127)


# ═══════════════════════════════════════════════════════════════════════
# Overflow Proof Tests
# ═══════════════════════════════════════════════════════════════════════


class TestOverflowProof:
    """Test formal overflow proofs via interval arithmetic."""

    def test_lif_safe_at_q88(self) -> None:
        """LIF derivative should be provably safe at Q8.8 with bounded inputs."""
        result = prove_no_overflow(
            "-(v - v_rest) / tau_m + R * I / C",
            bounds={
                "v": (-128, 127),
                "v_rest": (-65, -65),
                "tau_m": (10, 10),
                "R": (1, 1),
                "I": (0, 100),
                "C": (1, 1),
            },
            data_width=16,
            fraction=8,
        )
        assert result.proven_safe, (
            f"LIF should be safe at Q8.8: "
            f"result=[{result.expr_interval.lo:.1f}, {result.expr_interval.hi:.1f}]"
        )

    def test_overflow_detected(self) -> None:
        """Detect overflow when values exceed Q1.7 range."""
        result = prove_no_overflow(
            "v + I",
            bounds={"v": (-65, 30), "I": (0, 100)},
            data_width=8,
            fraction=7,
        )
        assert not result.proven_safe

    def test_safe_normalised_model(self) -> None:
        """Normalised FHN model within operating range should be safe at Q4.12."""
        result = prove_no_overflow(
            "a * (v - v * v * v) - w + I",
            bounds={
                "a": (0.5, 0.5),
                "v": (-1.5, 1.5),
                "w": (-1.5, 1.5),
                "I": (0, 0.5),
            },
            data_width=16,
            fraction=12,
        )
        assert result.proven_safe

    def test_margin_values(self) -> None:
        """Margins should be positive when safe, negative when unsafe."""
        safe = prove_no_overflow(
            "a + b",
            bounds={"a": (0, 10), "b": (0, 10)},
            data_width=16,
            fraction=8,
        )
        assert safe.margin_lo > 0
        assert safe.margin_hi > 0

    def test_unsigned_format(self) -> None:
        """Unsigned format has min=0, larger max."""
        result = prove_no_overflow(
            "a + b",
            bounds={"a": (0, 100), "b": (0, 100)},
            data_width=16,
            fraction=8,
            signed=False,
        )
        assert result.q_min == 0.0
        assert result.proven_safe


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
        for kwargs in (
            {"total_bits": 0, "fractional_bits": 0},
            {"total_bits": 16, "fractional_bits": 16},
        ):
            try:
                prove_fixed_point_envelope([1], **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"invalid format accepted: {kwargs}")

        try:
            prove_fixed_point_envelope([])
        except ValueError as exc:
            assert "at least one" in str(exc)
        else:
            raise AssertionError("empty proof bounds were accepted")

    def test_non_integer_bound_codes_are_rejected(self) -> None:
        """Proof inputs must be integer Q-code bounds, not floats or bools."""
        for bad_code in (1.0, True):
            try:
                prove_fixed_point_envelope([bad_code])
            except TypeError as exc:
                assert "integer Q-code" in str(exc)
            else:
                raise AssertionError(f"non-integer bound accepted: {bad_code!r}")


# ═══════════════════════════════════════════════════════════════════════
# SVA Generation Tests
# ═══════════════════════════════════════════════════════════════════════


class TestSVAGeneration:
    """Test SystemVerilog Assertion generation."""

    def test_basic_sva(self) -> None:
        """Basic SVA should contain module, assertions, and covers."""
        sva = generate_sva(["v"], data_width=16, fraction=8)
        assert "module sc_equation_neuron_sva" in sva
        assert "a_no_overflow_v" in sva
        assert "c_spike_reachable" in sva
        assert "c_no_spike" in sva

    def test_multiple_state_vars(self) -> None:
        """SVA with multiple state variables has assertions for each."""
        sva = generate_sva(["v", "u"], data_width=16, fraction=8)
        assert "a_no_overflow_v" in sva
        assert "a_no_overflow_u" in sva
        assert "c_v_nonzero" in sva
        assert "c_u_nonzero" in sva

    def test_input_bounds(self) -> None:
        """Input assumptions should be generated when bounds are provided."""
        sva = generate_sva(
            ["v"],
            data_width=16,
            fraction=8,
            input_bounds={"I_t": (-1000, 25600)},
        )
        assert "m_I_t_bound" in sva
        assert "assume property" in sva

    def test_stability_check(self) -> None:
        """Stability assertions should be present."""
        sva = generate_sva(["v"], data_width=16, fraction=8)
        assert "a_v_not_stuck_max" in sva
        assert "[*100]" in sva

    def test_custom_module_name(self) -> None:
        """Custom module name should be used."""
        sva = generate_sva(
            ["v"],
            module_name="sc_lif_loihi",
            data_width=24,
            fraction=12,
        )
        assert "sc_lif_loihi_sva" in sva

    def test_unsigned_sva(self) -> None:
        """Unsigned format should not use $signed."""
        sva = generate_sva(["v"], data_width=16, fraction=8, signed=False)
        assert "65535" in sva  # unsigned max

    def test_bind_directive(self) -> None:
        """Should include a commented bind directive."""
        sva = generate_sva(["v"], module_name="sc_lif")
        assert "bind sc_lif" in sva

    def test_do254_header(self) -> None:
        """Should reference DO-254 / IEC 61508."""
        sva = generate_sva(["v"])
        assert "DO-254" in sva
        assert "IEC 61508" in sva
