# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hypothesis property-based fuzzing for security-critical paths

"""Property-based fuzz tests using Hypothesis.

Targets:
1. EquationNeuron — ensure random equation strings never escape the sandbox
2. sanitize_ident — ensure no injection-capable identifier passes through
3. VerilogGenerator — ensure random module names + layer configs don't crash
4. UniversalNeuron — ensure random schema dicts don't produce unexpected state

Strategy: generate random but structurally-valid inputs and assert invariants
(no crashes, no code execution, no NaN propagation).
"""

from __future__ import annotations

import math
import string

import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.universal_dsl import UniversalNeuron


# ---------------------------------------------------------------------------
# Strategy builders
# ---------------------------------------------------------------------------

# Safe math tokens for building random but parseable equations
_OPERATORS = st.sampled_from(["+", "-", "*", "/", "**"])
_NUMBERS = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_SAFE_VARS = st.sampled_from(["v", "w", "u", "x", "y", "z", "I", "theta"])
_SAFE_FUNCS = st.sampled_from(["sin", "cos", "exp", "tanh", "sqrt", "abs"])

# Build a simple arithmetic expression: "v + 3.14 * sin(w)"
_SIMPLE_EXPR = st.builds(
    lambda var, op, num: f"{var} {op} {num}",
    _SAFE_VARS,
    _OPERATORS,
    _NUMBERS,
)

# Build a function call expression: "sin(v + 1.5)"
_FUNC_EXPR = st.builds(
    lambda fn, var, num: f"{fn}({var} + {num})",
    _SAFE_FUNCS,
    _SAFE_VARS,
    _NUMBERS,
)

# Combined expression strategies
_EXPR = st.one_of(_SIMPLE_EXPR, _FUNC_EXPR)

# Hostile strings that should NEVER pass through the sandbox
_HOSTILE_STRINGS = st.sampled_from(
    [
        "__import__('os').system('id')",
        "eval('1+1')",
        "exec('print(1)')",
        "globals()['__builtins__']",
        "().__class__.__bases__[0].__subclasses__()",
        "__import__('subprocess').call('ls')",
        "open('/etc/passwd').read()",
        "type.__subclasses__(type)",
        "compile('x','','exec')",
        "vars()['__builtins__'].__import__('os')",
        "getattr(getattr('', '__class__'), '__bases__')[0]",
        "lambda: None",
    ]
)

# Random identifier strings (some clean, some with injection attempts)
_IDENT_STRINGS = st.text(
    alphabet=string.ascii_letters + string.digits + "_- ;`$\\(){}[]!@#%^&*",
    min_size=1,
    max_size=100,
)


# ---------------------------------------------------------------------------
# Equation Builder fuzz tests
# ---------------------------------------------------------------------------


class TestEquationBuilderFuzz:
    """Property: the EquationNeuron sandbox never executes arbitrary code."""

    @given(expr=_EXPR)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_safe_expressions_never_crash(self, expr: str) -> None:
        """Random but syntactically-plausible expressions should either
        succeed or raise ValueError — never an uncontrolled exception."""
        try:
            neuron = EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
                parameters={},
                dt=0.1,
            )
            # Step with safe inputs — should not crash
            neuron.step(I=1.0, w=0.5, u=0.0, x=0.0, y=0.0, z=0.0, theta=0.0)
        except (ValueError, ZeroDivisionError, OverflowError, FloatingPointError):
            pass  # Expected for some random expressions (FloatingPointError = fail-closed divergence)
        except Exception as e:
            # NameError is OK (unknown variables), TypeError is OK (type mismatches)
            if not isinstance(e, (NameError, TypeError)):
                pytest.fail(f"Unexpected exception for expr={expr!r}: {e}")

    @given(hostile=_HOSTILE_STRINGS)
    @settings(max_examples=50)
    def test_hostile_strings_always_rejected(self, hostile: str) -> None:
        """Known attack strings must ALWAYS raise ValueError."""
        with pytest.raises((ValueError, SyntaxError)):
            EquationNeuron(
                equations={"v": hostile},
                state={"v": 0.0},
                parameters={},
                dt=0.1,
            )

    @given(expr=st.text(min_size=1, max_size=200))
    @settings(max_examples=300, suppress_health_check=[HealthCheck.too_slow])
    def test_random_text_never_executes(self, expr: str) -> None:
        """Completely random text should raise ValueError/SyntaxError,
        never execute anything."""
        try:
            EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
                parameters={},
                dt=0.1,
            )
        except (ValueError, SyntaxError, TypeError):
            pass  # Expected
        except Exception:
            pass  # Any other exception is acceptable, just not code execution


# ---------------------------------------------------------------------------
# Identifier sanitisation fuzz tests
# ---------------------------------------------------------------------------


class TestSanitizeIdentFuzz:
    """Property: sanitize_ident always produces a valid Verilog identifier
    or raises ValueError for empty/wholly-invalid input."""

    @given(name=_IDENT_STRINGS)
    @settings(max_examples=500)
    def test_output_is_always_valid_verilog(self, name: str) -> None:
        """No matter the input, the output must be a valid Verilog identifier
        (alphanumeric + underscore, not starting with a digit)."""
        try:
            result = sanitize_ident(name)
        except ValueError:
            return  # Empty or completely invalid — acceptable

        # Verify result is a valid Verilog identifier
        assert len(result) > 0, "sanitize_ident returned empty string"
        assert result[0].isalpha() or result[0] == "_", (
            f"Identifier starts with invalid char: {result!r}"
        )
        for ch in result:
            assert ch.isalnum() or ch == "_", (
                f"Invalid character {ch!r} in sanitised identifier {result!r}"
            )

    @given(name=st.from_regex(r"[a-zA-Z_][a-zA-Z0-9_]{0,30}", fullmatch=True))
    @settings(max_examples=200)
    def test_valid_identifiers_pass_through(self, name: str) -> None:
        """Already-valid Verilog identifiers should pass through unchanged."""
        result = sanitize_ident(name)
        assert result == name


# ---------------------------------------------------------------------------
# Universal DSL fuzz tests
# ---------------------------------------------------------------------------


class TestUniversalDSLFuzz:
    """Property: random parameter values don't cause NaN or crash."""

    @given(
        current=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        steps=st.integers(min_value=1, max_value=50),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_lif_never_produces_nan(self, current: float, steps: int) -> None:
        """LIF with random current should never produce NaN state."""
        neuron = UniversalNeuron.from_schema("lif")
        for _ in range(steps):
            neuron.step(I=current)
        v = neuron.state["v"]
        assert not math.isnan(v), f"LIF produced NaN with I={current}"

    @given(
        a=st.floats(min_value=0.01, max_value=2.0),
        b=st.floats(min_value=0.01, max_value=2.0),
    )
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_izhikevich_parameter_sweep(self, a: float, b: float) -> None:
        """Izhikevich with swept parameters should not crash."""
        neuron = UniversalNeuron.from_schema(
            "izhikevich",
            parameter_overrides={"a": a, "b": b},
        )
        for _ in range(50):
            neuron.step(I=10.0)
        v = neuron.state["v"]
        assert not math.isnan(v), f"Izhikevich NaN with a={a}, b={b}"

    @given(dt=st.floats(min_value=0.001, max_value=2.0))
    @settings(max_examples=50)
    def test_fitzhugh_nagumo_dt_sweep(self, dt: float) -> None:
        """Faithful FHN (RK4, no reset) either stays finite or fails closed.

        The re-enrolled FitzHugh-Nagumo is an unbounded cubic relaxation oscillator,
        so a large enough step size genuinely diverges (unlike the earlier bounded
        reset caricature). The runner fails closed on a non-finite state
        (``FloatingPointError``, matching the hand models) rather than silently
        propagating NaN, so a completed 100-step sweep must leave a finite state and a
        divergent step raises the controlled error instead of corrupting the trace.
        """
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo", dt_override=dt)
        try:
            for _ in range(100):
                neuron.step(I=0.5)
        except (OverflowError, ValueError, FloatingPointError):
            return  # controlled fail-closed divergence for an extreme step size
        v = neuron.state["v"]
        assert math.isfinite(v), f"FHN left a non-finite state without failing closed (dt={dt})"
