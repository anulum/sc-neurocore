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
    Interval,
    compute_guard_bits,
    compute_guard_bits_multi,
    generate_sva,
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
