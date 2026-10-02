# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored equation admission and stochastic-state guards

"""Exercise deliberate constructor refusals and actual stochastic rollback."""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.equation_refusals import EquationRefusal, EquationStateFailure
from sc_neurocore.refusals import AuthoredRefusal


@pytest.mark.parametrize(
    ("method", "detection", "rate", "probability", "dt", "reason"),
    [
        ("bad", "level", None, None, 0.1, "method must be one of"),
        ("euler", "bad", None, None, 0.1, "detection must be one of"),
        ("euler", "level", "1", None, 0.1, "rate_expression is only valid"),
        ("euler", "level", None, "0.5", 0.1, "probability_expression is only valid"),
        ("euler", "poisson", None, None, 0.1, "poisson detection requires"),
        ("euler", "escape_rate", "1", None, 0.0, "escape_rate dt must be"),
        ("euler", "escape_rate", "1", None, -0.1, "escape_rate dt must be"),
        ("euler", "escape_rate", "1", None, float("inf"), "escape_rate dt must be"),
    ],
)
def test_constructor_admission_keeps_deliberate_reasons_and_valueerror_compatibility(
    method: str,
    detection: str,
    rate: str | None,
    probability: str | None,
    dt: float,
    reason: str,
) -> None:
    """Refuse incompatible methods, event fields and invalid stochastic timebases."""
    with pytest.raises(ValueError) as refused:
        EquationNeuron(
            equations={"v": "0"},
            state={"v": 0.0},
            method=method,
            detection=detection,
            rate_expression=rate,
            probability_expression=probability,
            dt=dt,
        )
    assert isinstance(refused.value, EquationRefusal)
    assert str(refused.value).startswith(reason)


def test_previous_state_alias_cannot_be_shadowed_by_a_parameter() -> None:
    """Protect the actual previous-state namespace used by event evaluation."""
    with pytest.raises(EquationRefusal) as refused:
        EquationNeuron(equations={"v": "0"}, parameters={"v_prev": 1.0}, state={"v": 0.0})
    assert str(refused.value) == (
        "Names reserved for previous-state aliases cannot be declared: v_prev"
    )


@pytest.mark.parametrize(
    ("detection", "rate", "probability", "dt", "reason"),
    [
        ("escape_rate", "-1", None, 0.1, "escape rate must remain finite and non-negative"),
        (
            "escape_rate",
            "exp(700)",
            None,
            1e308,
            "escape hazard must remain finite and non-negative",
        ),
        ("poisson", None, "2", 0.1, "stochastic spike probability must remain finite and bounded"),
    ],
)
def test_actual_stochastic_guard_restores_state_and_rng(
    detection: str, rate: str | None, probability: str | None, dt: float, reason: str
) -> None:
    """Refuse invalid measured rates/probabilities without consuming a trial."""
    neuron = EquationNeuron(
        equations={"v": "0"},
        state={"v": 0.0},
        detection=detection,
        rate_expression=rate,
        probability_expression=probability,
        dt=dt,
    )
    rng_before = neuron.stochastic_rng_state
    with pytest.raises(FloatingPointError) as refused:
        neuron.step()
    assert isinstance(refused.value, EquationStateFailure)
    assert str(refused.value) == reason
    assert neuron.state == {"v": 0.0}
    assert neuron.stochastic_rng_state == rng_before


def test_generated_constructor_conversion_is_not_marked_as_authored() -> None:
    """Keep a real malformed float conversion out of the trusted reason class."""
    with pytest.raises(ValueError) as failure:
        EquationNeuron(
            equations={"v": "0"},
            detection="poisson",
            probability_expression="0.5",
            dt="caller-text-xyz",
        )
    assert not isinstance(failure.value, AuthoredRefusal)
