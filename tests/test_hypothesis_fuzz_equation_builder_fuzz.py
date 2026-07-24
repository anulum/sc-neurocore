# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestEquationBuilderFuzz from former test_hypothesis_fuzz.py

"""Focused suite: TestEquationBuilderFuzz from former test_hypothesis_fuzz.py."""

from __future__ import annotations

from tests.hypothesis_fuzz_support import *  # noqa: F403


class TestEquationBuilderFuzz:
    """Property: the EquationNeuron sandbox never executes arbitrary code."""

    @given(expr=_EXPR)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_safe_expressions_fail_closed_or_produce_finite_state(self, expr: str) -> None:
        """Random but syntactically-plausible expressions should either
        succeed or raise ValueError — never an uncontrolled exception."""
        try:
            neuron = EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
                parameters={},
                dt=0.1,
            )
            with np.errstate(divide="raise", invalid="raise", over="raise"):
                neuron.step(I=1.0, w=0.5, u=0.0, x=0.0, y=0.0, z=0.0, theta=0.0)
        except (
            ValueError,
            ZeroDivisionError,
            OverflowError,
            FloatingPointError,
            NameError,
            TypeError,
        ):
            return
        except Exception as exc:
            pytest.fail(f"Unexpected exception for expr={expr!r}: {exc}")

        assert math.isfinite(float(neuron.state["v"]))

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
    def test_random_text_fails_closed_without_uncontrolled_exceptions(self, expr: str) -> None:
        """Random text is either rejected or produces a valid sandboxed neuron."""
        try:
            neuron = EquationNeuron(
                equations={"v": expr},
                state={"v": 0.0},
                parameters={},
                dt=0.1,
            )
        except (ValueError, SyntaxError, TypeError):
            return
        except Exception as exc:
            pytest.fail(f"Unexpected exception for random expression {expr!r}: {exc}")

        assert isinstance(neuron, EquationNeuron)
