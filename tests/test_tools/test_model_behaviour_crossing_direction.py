# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Every crossing-count model states a case that tells an upward crossing from a downward one

"""A model whose event is an upward crossing states a case that a downward count would break.

Some case models report an event when one state variable goes from below a
threshold to at or above it, and nothing resets it. A window that ends where the
variable has fallen back below the threshold holds as many downward crossings
as upward ones, so a step that counted the other direction would still give the
stated count. These tests run every spike case of such a model through its
public class, count the crossings of that variable in both directions, and
require one case whose upward count is the larger. They also require every case model to
be classified, so that a new crossing model is held to the rule on purpose.
"""

from __future__ import annotations

from functools import cache
import math

import pytest

from sc_neurocore.neurons import models as _models
from sc_neurocore.neurons.model_catalogue import load_descriptor
from sc_neurocore.neurons.model_descriptor import BEHAVIOUR_SPIKE_OBSERVABLE
from sc_neurocore.neurons.models import _CLASS_TO_MODULE

#: Case models whose step returns 1 when a state variable crosses a threshold
#: upwards, with no reset: the variable and the threshold attribute the step
#: compares. A model moves here or to RESET_OR_LEVEL_MODELS only by a deliberate
#: change in the unit that gives it cases.
CROSSING_MODELS: dict[str, tuple[str, str]] = {
    "ChialvoMapNeuron": ("x", "x_threshold"),
    "CourageNekorkinMapNeuron": ("x", "x_threshold"),
    "ErmentroutKopellMapNeuron": ("theta", "theta_threshold"),
    "FitzHughNagumoNeuron": ("v", "v_threshold"),
    "FitzHughRinzelNeuron": ("v", "v_threshold"),
    "HindmarshRoseNeuron": ("x", "x_threshold"),
    "PernarowskiNeuron": ("v", "v_threshold"),
    "SCAdaptiveThresholdMapNeuron": ("x", "x_threshold"),
    "SCUpwardCrossingRulkovMapNeuron": ("x", "x_threshold"),
    "WilsonHRNeuron": ("v", "v_peak"),
}

#: The period of a variable the class stores wrapped. The Ermentrout-Kopell
#: phase is kept in [0, 2π), so a crossing is counted on the unwrapped phase,
#: against the threshold and its images one period apart.
PERIODIC: dict[str, float] = {"ErmentroutKopellMapNeuron": 2.0 * math.pi}

#: Case models whose event is a reset, a branch of the map or a level, where a
#: downward crossing is not a natural fault of the step.
RESET_OR_LEVEL_MODELS = frozenset(
    {
        "AdaptiveThresholdIFNeuron",
        "DendriticNMDANeuron",
        "ExpIFNeuron",
        "IbarzTanakaMapNeuron",
        "IntegerQIFNeuron",
        "MihalasNieburNeuron",
        "RulkovMapNeuron",
        "SCClippedLogisticBurstingMapNeuron",
        "SCFourStateGLIFNeuron",
        "SCInclusivePerfectIntegratorNeuron",
        "SCResettingWilsonHRNeuron",
        "SCSymmetricQuadraticIFNeuron",
        "SigmaDeltaNeuron",
    }
)


def _models_with_cases() -> set[str]:
    """Return the catalogue models whose descriptor states at least one case."""
    named = set()
    for name in _CLASS_TO_MODULE:
        descriptor = load_descriptor(name)
        if descriptor is not None and descriptor.behaviour:
            named.add(name)
    return named


@cache
def _crossings(model: str) -> tuple[tuple[str, int, int, int, int], ...]:
    """Run each spike case of a crossing model and count its events and crossings.

    Parameters
    ----------
    model : str
        A name of :data:`CROSSING_MODELS`.

    Returns
    -------
    tuple of (str, int, int, int, int)
        Per spike case, in the descriptor's order: its name, the events the step
        returned, the upward and the downward crossings of the variable, and the
        steps where the step's event and an upward crossing do not coincide.
    """
    variable, threshold_name = CROSSING_MODELS[model]
    period = PERIODIC.get(model)
    descriptor = load_descriptor(model)
    assert descriptor is not None
    rows = []
    for case in descriptor.behaviour:
        if case.observable != BEHAVIOUR_SPIKE_OBSERVABLE:
            continue
        neuron = getattr(_models, model)(**dict(case.parameters))
        threshold = float(getattr(neuron, threshold_name))
        before = float(getattr(neuron, variable))
        offset = 0.0
        events = ups = downs = mismatched = 0
        for _ in range(case.steps):
            event = int(neuron.step(**dict(case.inputs)))
            raw = float(getattr(neuron, variable))
            if period is None:
                after = raw
                up = int(before < threshold <= after)
                down = int(after < threshold <= before)
            else:
                # A jump of more than half a period is the class wrapping the stored phase.
                if raw + offset < before - period / 2.0:
                    offset += period
                elif raw + offset > before + period / 2.0:
                    offset -= period
                after = raw + offset
                level_after = math.floor((after - threshold) / period)
                level_before = math.floor((before - threshold) / period)
                up = max(level_after - level_before, 0)
                down = max(level_before - level_after, 0)
            events += event
            ups += up
            downs += down
            mismatched += int(event != min(up, 1))
            before = after
        rows.append((case.name, events, ups, downs, mismatched))
    return tuple(rows)


def test_every_case_model_is_classified() -> None:
    """A case model is either a crossing model or a reset or level model, never both."""
    assert set(CROSSING_MODELS).isdisjoint(RESET_OR_LEVEL_MODELS)
    assert set(CROSSING_MODELS) | RESET_OR_LEVEL_MODELS == _models_with_cases()
    assert set(PERIODIC) <= set(CROSSING_MODELS)


@pytest.mark.parametrize("model", sorted(CROSSING_MODELS))
def test_the_steps_events_are_the_upward_crossings_this_test_counts(model: str) -> None:
    """The table names the variable and threshold the step compares: each event is an upward crossing.

    The events and the upward crossings must fall on the same steps, not only
    agree in number.
    """
    rows = _crossings(model)
    assert rows, f"{model} states no spike case"
    for name, events, ups, _downs, mismatched in rows:
        assert (events, mismatched) == (ups, 0), (
            f"{model}: {name!r} returned {events} events and {ups} upward crossings, "
            f"apart on {mismatched} steps"
        )


@pytest.mark.parametrize("model", sorted(CROSSING_MODELS))
def test_a_stated_case_tells_the_crossing_direction(model: str) -> None:
    """One spike case ends with more upward crossings than downward ones, so a downward count breaks it."""
    rows = _crossings(model)
    telling = [name for name, _events, ups, downs, _mismatched in rows if ups > downs]
    assert telling, (
        f"{model}: every spike case holds as many downward crossings as upward ones: "
        + ", ".join(f"{name!r} {ups} and {downs}" for name, _events, ups, downs, _m in rows)
    )
