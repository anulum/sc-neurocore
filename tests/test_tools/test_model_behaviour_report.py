# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The behaviour report runs the cases a descriptor states and says what each showed

"""Run behaviour cases through real models and bind what the report says about them.

The models of these tests are small real integrators, stepped as any catalogue
model is. A case that holds only where nothing happens is reported as held
vacuously, a case that fails is reported with what was observed, and a case
that cannot be run says why. The report states each case beside its result and
binds the stated cases of a model by one digest.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
import re
import runpy
import sys

import numpy as np
import pytest

from sc_neurocore.neurons.model_descriptor import BehaviourCase
from sc_neurocore.studio.candidate_package import MAX_CANDIDATE_STEPS
from tools.model_behaviour_report import (
    DEFAULT_OUTPUT,
    MAX_CASE_STEPS,
    MODEL_STATES,
    REPORT_SCHEMA,
    REPORT_NOTES,
    build_report,
    case_holds,
    cases_digest,
    check_problems,
    encode,
    evaluate_model,
    main,
    render_summary,
    report_of,
)

ROOT = Path(__file__).resolve().parents[2]

#: Models of the catalogue that state no behaviour case. It moves only by a
#: deliberate change here, in the unit that adds or removes cases.
MODELS_WITHOUT_A_CASE = 160

#: Every case's stated limit, in the descriptor's order: an interval's bounds or a
#: contrast's least margin. The observed counts alone do not hold a case's claim: a
#: bound widened in the descriptor, with both reports written again, would still pass.
STATED_LIMITS: dict[str, list[object]] = {
    "AdaptiveThresholdIFNeuron": [[1, 1], [3, 3]],
    "AiharaMapNeuron": [[6, 6], [4, 4], [2, 2]],
    "ChialvoMapNeuron": [[3, 3], [1, 1]],
    "CourageNekorkinMapNeuron": [[0, 0], [0, 0], [1, 1], [1, 1], [3, 3], [0.5, 0.625]],
    "DendriticNMDANeuron": [
        [0, 0],
        [0, 0],
        [-60.0, -59.0],
        [50, 70],
        [180, 200],
        [200, 220],
        10,
        [266, 266],
    ],
    "ErmentroutKopellMapNeuron": [[0, 0], [45, 45], [64, 64], [5, 5]],
    "ExpIFNeuron": [[2, 2], [9, 9]],
    "FitzHughNagumoNeuron": [[0, 1], [26, 26], [28, 28], [28, 28], [1, 1]],
    "FitzHughRinzelNeuron": [[7, 7], [8, 8], [8, 8], [83, 83], [1, 1]],
    "HindmarshRoseNeuron": [[0, 0], [0, 0], [26, 26], [40, 40], [52, 52], [1, 1]],
    "IbarzTanakaMapNeuron": [[0, 0], [3, 3], [7, 7], [18, 18]],
    "IntegerQIFNeuron": [[26, 26]],
    "MihalasNieburNeuron": [[8, 8], [14, 14]],
    "NagumoSatoMapNeuron": [[1, 1], [4, 4], [12, 12]],
    "PernarowskiNeuron": [[17, 17], [17, 17], [17, 17], [17, 17], [54, 54], [1, 1]],
    "RulkovMapNeuron": [[0, 0], [7, 7], [10, 10], [0, 0], [34, 34], [130, 130]],
    "SCAdaptiveThresholdMapNeuron": [[3, 3], [4, 4]],
    "SCClippedLogisticBurstingMapNeuron": [[2, 2], [1, 1], [1, 1], [7, 7]],
    "SCFourStateGLIFNeuron": [[0, 0], [0, 0], [23, 23], [54, 54], [86, 86], [95, 95]],
    "SCInclusivePerfectIntegratorNeuron": [[500, 500], [500, 500]],
    "SCResettingWilsonHRNeuron": [[0, 0], [1, 1], [4, 4]],
    "SCSymmetricQuadraticIFNeuron": [
        [0, 0],
        [2, 2],
        [3, 3],
        [6, 6],
        [11, 11],
        [26, 26],
        [100, 100],
        [250, 250],
    ],
    "SCUpwardCrossingRulkovMapNeuron": [[0, 0], [7, 7], [10, 10], [11, 11], [11, 11]],
    "SigmaDeltaNeuron": [[42, 42]],
    "WilsonHRNeuron": [[0, 0], [44, 44], [46, 46], [49, 49], [52, 52], [1, 1]],
}


class _Integrator:
    """Accumulate an eighth of the current each step; fire and reset when the level reaches one.

    The step size is a power of two, so the level is exact and the model fires
    on every eighth step at unit current.
    """

    def __init__(self) -> None:
        self.level = 0.0

    def step(self, current: float) -> int:
        """Advance one step and return 1 on a spike."""
        self.level += 0.125 * current
        if self.level >= 1.0:
            self.level = 0.0
            return 1
        return 0


class _TwoInputs:
    """Fire only when a gate is open and the drive is strong; keep a trace of the gate."""

    def __init__(self) -> None:
        self.trace = -1.0

    def step(self, drive: float, gate: float, bias: float = 0.0) -> int:
        """Advance one step; the trace relaxes towards the gate."""
        self.trace += 0.5 * (gate - self.trace)
        return 1 if gate * drive + bias >= 1.0 else 0


class _Refuses:
    """A model whose step refuses a negative current."""

    def step(self, current: float) -> int:
        """Refuse a negative drive, as catalogue models refuse invalid input."""
        if current < 0:
            raise ValueError("current must not be negative")
        return 0


class _Rate:
    """A model whose step returns a rate, not a spike, and keeps it as its state."""

    def __init__(self) -> None:
        self.rate = 0.0

    def step(self, current: float) -> float:
        """Return half the drive."""
        self.rate = 0.5 * current
        return self.rate


class _IndexedByItsInput:
    """A model whose step needs a whole number: it uses its input as an index."""

    def step(self, channel: int) -> int:
        """Return the entry of a table at the given channel."""
        return (0, 1, 0)[channel]


class _ArrayScalarState:
    """A model that keeps its state as a numpy scalar, as vectorised model code does."""

    def __init__(self) -> None:
        self.level = np.float64(0.0)

    def step(self, current: float) -> int:
        """Add the current to the level."""
        self.level = self.level + np.float64(current)
        return 0


class _FailsInItsOwnWay:
    """A model whose step raises an error of its own kind."""

    def step(self, current: float) -> int:
        """Raise what a model's internal check might raise."""
        raise RuntimeError("the solver did not converge")


class _TruthValueSpikes:
    """A model whose step returns a numpy truth value, as a vectorised threshold test does."""

    def __init__(self) -> None:
        self.level = 0.0

    def step(self, current: float) -> np.bool_:
        """Fire on every second step at unit current."""
        self.level += 0.5 * current
        fired = np.float64(self.level) >= 1.0
        if fired:
            self.level = 0.0
        return fired


class _TakesAnything:
    """A model whose step takes any named input and fires when their sum reaches one."""

    def step(self, **inputs: float) -> int:
        """Fire when the inputs sum to one or more."""
        return 1 if sum(inputs.values()) >= 1.0 else 0


class _NeedsAnArgument:
    """A model that cannot be built without an argument."""

    def __init__(self, size: int) -> None:
        self.size = size

    def step(self, current: float) -> int:
        """Never fire."""
        return 0


class _Gain:
    """Accumulate the current times a gain each step; fire and reset when the level reaches one.

    The gain is a constructor parameter, as a catalogue model's parameters are,
    and a negative gain is refused, as a model's own validation refuses a value.
    By default a unit current fires on every fourth step.
    """

    def __init__(self, gain: float = 0.25) -> None:
        if gain < 0:
            raise ValueError("gain must not be negative")
        self.gain = gain
        self.level = 0.0

    def step(self, current: float) -> int:
        """Advance one step and return 1 on a spike."""
        self.level += self.gain * current
        if self.level >= 1.0:
            self.level = 0.0
            return 1
        return 0


def _interval(
    name: str,
    inputs: dict[str, float],
    bounds: tuple[float, float],
    *,
    observable: str = "spikes",
    steps: int = 100,
    parameters: dict[str, float] | None = None,
) -> BehaviourCase:
    return BehaviourCase(
        name=name,
        kind="interval",
        steps=steps,
        observable=observable,
        inputs=tuple(sorted(inputs.items())),
        bounds=bounds,
        parameters=tuple(sorted((parameters or {}).items())),
    )


def _contrast(
    name: str, inputs: dict[str, float], reference: dict[str, float], margin: float
) -> BehaviourCase:
    return BehaviourCase(
        name=name,
        kind="contrast",
        steps=100,
        observable="spikes",
        inputs=tuple(sorted(inputs.items())),
        reference_inputs=tuple(sorted(reference.items())),
        margin=margin,
    )


def test_a_case_that_holds_where_the_model_is_active_is_held() -> None:
    """Twelve spikes in a hundred steps at unit current: observed, held and exercised."""
    result = evaluate_model(_Integrator, (_interval("fires", {"current": 1.0}, (12, 12)),))
    assert result["state"] == "held"
    assert result["cases"] == [
        {
            "name": "fires",
            "kind": "interval",
            "observable": "spikes",
            "steps": 100,
            "inputs": {"current": 1.0},
            "bounds": [12, 12],
            "observed": 12,
            "held": True,
            "exercised": True,
        }
    ]


def test_a_case_that_fails_is_reported_with_what_was_observed() -> None:
    """The observed count stands beside the verdict, so a reader sees how far off it is."""
    result = evaluate_model(_Integrator, (_interval("fires more", {"current": 1.0}, (20, 30)),))
    assert result["state"] == "failed"
    assert result["cases"][0]["observed"] == 12
    assert result["cases"][0]["held"] is False


def test_an_observation_above_the_upper_end_fails() -> None:
    """Both ends bind: twelve spikes do not lie within one to five."""
    result = evaluate_model(_Integrator, (_interval("fires rarely", {"current": 1.0}, (1, 5)),))
    assert result["state"] == "failed"
    assert result["cases"][0]["observed"] == 12


def test_cases_that_hold_only_at_rest_are_held_vacuously() -> None:
    """A model that does nothing at all satisfies every such case: that is not evidence."""
    cases = (
        _interval("silent without drive", {"current": 0.0}, (0, 0)),
        _interval("level stays at rest", {"current": 0.0}, (0.0, 0.0), observable="state:level"),
    )
    result = evaluate_model(_Integrator, cases)
    assert result["state"] == "held_vacuously"
    assert [case["held"] for case in result["cases"]] == [True, True]
    assert [case["exercised"] for case in result["cases"]] == [False, False]


def test_one_exercised_case_makes_the_zero_cases_count() -> None:
    """Exact zeros are kept; they count because another case observes activity."""
    cases = (
        _interval("silent without drive", {"current": 0.0}, (0, 0)),
        _interval("fires", {"current": 1.0}, (12, 12)),
    )
    assert evaluate_model(_Integrator, cases)["state"] == "held"


def test_a_state_observable_is_the_final_value_and_is_exercised_when_it_moved() -> None:
    """The trace relaxes from -1 to the gate; its final value is read after the last step."""
    case = _interval(
        "trace follows the gate", {"drive": 0.0, "gate": 1.0}, (0.99, 1.0), observable="state:trace"
    )
    (reported,) = evaluate_model(_TwoInputs, (case,))["cases"]
    assert reported["observed"] == pytest.approx(1.0, abs=1e-6)
    assert reported["held"] is True
    assert reported["exercised"] is True


def test_a_contrast_holds_by_its_margin_and_reports_both_observations() -> None:
    """With the gate open the model fires every step; with it closed, never."""
    case = _contrast(
        "the gate matters", {"drive": 2.0, "gate": 1.0}, {"drive": 2.0, "gate": 0.0}, margin=50
    )
    (reported,) = evaluate_model(_TwoInputs, (case,))["cases"]
    assert reported["inputs"] == {"drive": 2.0, "gate": 1.0}
    assert reported["reference_inputs"] == {"drive": 2.0, "gate": 0.0}
    assert reported["margin"] == 50
    assert "bounds" not in reported
    assert reported["observed"] == 100
    assert reported["reference_observed"] == 0
    assert reported["held"] is True
    assert reported["exercised"] is True


def test_a_contrast_below_its_margin_fails() -> None:
    """A difference smaller than the margin is a failure, not a pass by order alone."""
    case = _contrast(
        "the gate matters a lot",
        {"drive": 2.0, "gate": 1.0},
        {"drive": 2.0, "gate": 0.0},
        margin=101,
    )
    result = evaluate_model(_TwoInputs, (case,))
    assert result["state"] == "failed"
    assert result["cases"][0]["held"] is False
    assert result["cases"][0]["exercised"] is False


def test_a_contrast_that_meets_its_margin_exactly_holds() -> None:
    """The margin is the least difference: a difference equal to it holds."""
    case = _contrast(
        "the gate decides every step",
        {"drive": 2.0, "gate": 1.0},
        {"drive": 2.0, "gate": 0.0},
        margin=100,
    )
    assert evaluate_model(_TwoInputs, (case,))["state"] == "held"


def test_a_contrast_whose_reference_drive_cannot_be_run_reports_no_observation() -> None:
    """The first drive ran; the case is still invalid and no half result is written."""
    case = _contrast("x", {"current": 1.0}, {"current": -1.0}, margin=1)
    result = evaluate_model(_Refuses, (case,))
    assert result["state"] == "invalid"
    assert "observed" not in result["cases"][0]
    assert result["cases"][0]["reason"] == "the step raised ValueError"


def test_a_final_state_value_is_written_with_six_significant_digits() -> None:
    """After ten steps the trace is 1 - 2 / 1024 = 0.998046875, written as 0.998047."""
    case = _interval(
        "trace after ten steps",
        {"drive": 0.0, "gate": 1.0},
        (0.99, 1.0),
        observable="state:trace",
        steps=10,
    )
    assert evaluate_model(_TwoInputs, (case,))["cases"][0]["observed"] == 0.998047


def test_a_step_that_takes_any_named_input_can_be_driven_by_any_name() -> None:
    """Where the step declares that it takes any keyword, no input name is refused."""
    case = _interval("two halves fire", {"left": 0.5, "right": 0.5}, (100, 100))
    assert evaluate_model(_TakesAnything, (case,))["state"] == "held"


def test_an_optional_input_may_be_given_or_left_out() -> None:
    """A step argument with a default is an input a case may name."""
    case = _interval("bias alone fires", {"drive": 0.0, "gate": 0.0, "bias": 1.0}, (100, 100))
    assert evaluate_model(_TwoInputs, (case,))["state"] == "held"


_INVALID = {
    "an input the step does not take": (
        _TwoInputs,
        _interval("x", {"drive": 1.0, "gate": 1.0, "glutamate": 1.0}, (1, 100)),
        "the step takes no input named ['glutamate']",
    ),
    "a required input left out": (
        _TwoInputs,
        _interval("x", {"drive": 1.0}, (1, 100)),
        "the step requires the input ['gate']",
    ),
    "a state variable the model does not hold": (
        _Integrator,
        _interval("x", {"current": 1.0}, (0.0, 1.0), observable="state:calcium"),
        "the model holds no state variable named 'calcium'",
    ),
    "a step that refuses the drive": (
        _Refuses,
        _interval("x", {"current": -1.0}, (0, 0)),
        "the step raised ValueError",
    ),
    "a spike count of a step that returns a rate": (
        _Rate,
        _interval("x", {"current": 1.0}, (1, 100)),
        "the step returned a value that is not 0 or 1",
    ),
    "a step that fails in its own way": (
        _FailsInItsOwnWay,
        _interval("x", {"current": 1.0}, (1, 100)),
        "the step raised RuntimeError",
    ),
    "a step that needs a whole number": (
        _IndexedByItsInput,
        _interval("x", {"channel": 1.0}, (1, 100)),
        "the step raised TypeError",
    ),
    "a model that cannot be built without arguments": (
        _NeedsAnArgument,
        _interval("x", {"current": 1.0}, (0, 0)),
        "the model cannot be built without arguments",
    ),
}


@pytest.mark.parametrize("reason", sorted(_INVALID))
def test_a_case_that_cannot_be_run_is_invalid_and_says_why(reason: str) -> None:
    """Nothing is observed and nothing is held; the reason is the whole sentence of the fault."""
    model, case, message = _INVALID[reason]
    result = evaluate_model(model, (case,))
    assert result["state"] == "invalid"
    (reported,) = result["cases"]
    assert reported["held"] is False
    assert "observed" not in reported
    assert reported["reason"] == message


def test_a_state_variable_of_a_model_that_returns_a_rate_can_be_bounded() -> None:
    """What the step returns is read for a spike count only; a state case does not need it."""
    case = _interval(
        "the rate is half the drive", {"current": 3.0}, (1.5, 1.5), observable="state:rate"
    )
    result = evaluate_model(_Rate, (case,))
    assert result["state"] == "held"
    assert result["cases"][0]["observed"] == 1.5
    assert result["cases"][0]["bounds"] == [1.5, 1.5]


def test_a_truth_value_returned_by_a_step_counts_as_a_spike() -> None:
    """A numpy truth value is 0 or 1: fifty spikes in a hundred steps are counted."""
    result = evaluate_model(_TruthValueSpikes, (_interval("x", {"current": 1.0}, (50, 50)),))
    assert result["state"] == "held"
    assert result["cases"][0]["observed"] == 50
    assert type(result["cases"][0]["observed"]) is int


def test_a_case_holds_at_both_ends_of_its_interval_and_at_its_margin() -> None:
    """One rule for both tools: ends are included, and a difference equal to the margin holds."""
    interval = _interval("x", {"current": 1.0}, (12, 14))
    assert [case_holds(interval, value) for value in (11, 12, 14, 15)] == [False, True, True, False]
    contrast = _contrast("x", {"drive": 2.0}, {"drive": 1.0}, margin=10)
    assert case_holds(contrast, 30, 20) is True
    assert case_holds(contrast, 29, 20) is False
    assert case_holds(contrast, 20, 30) is False


def test_a_bound_is_written_exactly_as_stated() -> None:
    """Only an observation is shortened to six digits; a stated bound or margin is not."""
    case = _interval(
        "the rate is half the drive",
        {"current": 3.0},
        (1.4999999999, 1.5000000001),
        observable="state:rate",
    )
    (reported,) = evaluate_model(_Rate, (case,))["cases"]
    assert reported["bounds"] == [1.4999999999, 1.5000000001]
    assert reported["held"] is True
    margin = BehaviourCase(
        name="more drive, more rate",
        kind="contrast",
        steps=1,
        observable="state:rate",
        inputs=(("current", 3.0),),
        reference_inputs=(("current", 1.0),),
        margin=0.9999999999,
    )
    (reported,) = evaluate_model(_Rate, (margin,))["cases"]
    assert reported["margin"] == 0.9999999999
    assert reported["held"] is True
    spike_margin = evaluate_model(
        _TwoInputs,
        (_contrast("x", {"drive": 2.0, "gate": 1.0}, {"drive": 2.0, "gate": 0.0}, margin=50.0),),
    )["cases"][0]["margin"]
    assert spike_margin == 50 and type(spike_margin) is int


def _as_digested(case: BehaviourCase) -> dict[str, object]:
    """Return a case as the digest reads it: every field, the parameters only when it sets some."""
    return {
        field: value
        for field, value in dataclasses.asdict(case).items()
        if field != "parameters" or value
    }


def test_the_stated_cases_of_a_model_are_bound_by_one_digest() -> None:
    """The digest is of the cases as stated, in order; any change of a statement changes it.

    A case that sets no parameter is digested without the field, so the digest
    of every case written before parameters could be set is what it was.
    """
    first = _interval("silent without drive", {"current": 0.0}, (0, 0))
    second = _interval("fires", {"current": 1.0}, (12, 12))
    result = evaluate_model(_Integrator, (first, second))
    canonical = json.dumps(
        [_as_digested(first), _as_digested(second)],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert "parameters" not in canonical
    assert result["cases_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert result["cases_sha256"] == cases_digest((first, second))
    assert cases_digest((second, first)) != result["cases_sha256"]
    for changed in (
        dataclasses.replace(second, bounds=(12.0, 13.0)),
        dataclasses.replace(second, steps=101),
        dataclasses.replace(second, inputs=(("current", 1.5),)),
        dataclasses.replace(second, name="fires twelve times"),
        dataclasses.replace(second, parameters=(("gain", 0.125),)),
    ):
        assert cases_digest((first, changed)) != result["cases_sha256"]


def test_a_state_kept_as_a_numpy_scalar_is_reported_in_plain_values() -> None:
    """The entry holds a plain number and plain truth values, so the report can be written."""
    case = _interval("level rises", {"current": 0.5}, (1.0, 1.0), observable="state:level", steps=2)
    result = evaluate_model(_ArrayScalarState, (case,))
    (reported,) = result["cases"]
    assert (reported["observed"], reported["held"], reported["exercised"]) == (1.0, True, True)
    assert type(reported["exercised"]) is bool
    assert type(reported["observed"]) is float
    assert json.loads(encode(report_of([{"model": "A", **result}])))["totals"]["held"] == 1


def test_a_case_may_run_as_many_steps_as_a_candidate_run_and_no_more() -> None:
    """The bound is the Studio's, so a case can also be run on a variant of the schema."""
    assert MAX_CASE_STEPS == MAX_CANDIDATE_STEPS
    at_the_bound = _interval("silent", {"current": 0.0}, (0, 0), steps=MAX_CASE_STEPS)
    assert evaluate_model(_Integrator, (at_the_bound,))["cases"][0]["held"] is True
    above = _interval("silent", {"current": 0.0}, (0, 0), steps=MAX_CASE_STEPS + 1)
    result = evaluate_model(_Integrator, (above,))
    assert result["state"] == "invalid"
    assert f"more than {MAX_CASE_STEPS} steps" in result["cases"][0]["reason"]


def test_every_run_starts_from_a_new_model() -> None:
    """Two cases of one model do not share state: the second starts at rest."""
    cases = (
        _interval(
            "seven steps leave the level high",
            {"current": 1.0},
            (0.875, 0.875),
            observable="state:level",
            steps=7,
        ),
        _interval("one step does not fire", {"current": 1.0}, (0, 0), steps=1),
    )
    result = evaluate_model(_Integrator, cases)
    assert [case["held"] for case in result["cases"]] == [True, True]


@pytest.fixture(scope="module")
def report() -> dict[str, object]:
    """Build the report once from the shipped descriptors."""
    return build_report()


def _models(report: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = report["models"]
    assert isinstance(rows, list)
    return {str(row["model"]): row for row in rows}


@pytest.mark.parametrize("model_name", sorted(STATED_LIMITS))
def test_every_case_states_the_limit_this_test_pins(
    report: dict[str, object], model_name: str
) -> None:
    """A case's bounds or margin are its claim; a widened one fails even with the reports written again.

    Parameters
    ----------
    report : dict[str, object]
        The report as the tool computes it.
    model_name : str
        A model that states cases.
    """
    cases = _models(report)[model_name]["cases"]
    assert isinstance(cases, list)
    stated = [case["bounds"] if case["kind"] == "interval" else case["margin"] for case in cases]
    assert stated == STATED_LIMITS[model_name]


def test_the_pinned_limits_cover_every_model_that_states_cases(report: dict[str, object]) -> None:
    """A model that gains cases has its limits pinned here as well."""
    stating = sorted(name for name, row in _models(report).items() if row["state"] != "no_case")
    assert stating == sorted(STATED_LIMITS)


def test_the_committed_report_is_what_the_tool_computes(report: dict[str, object]) -> None:
    """A stale report cannot stay committed."""
    assert (ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8") == encode(report)
    assert report["schema_version"] == REPORT_SCHEMA
    assert report["notes"] == list(REPORT_NOTES)


def test_every_model_has_one_of_the_states(report: dict[str, object]) -> None:
    """The totals are the row counts, and a model without cases lists none."""
    models = _models(report)
    assert report["catalogue_models"] == len(models)
    totals = report["totals"]
    assert isinstance(totals, dict)
    assert set(totals) == set(MODEL_STATES)
    for state in MODEL_STATES:
        assert totals[state] == sum(1 for row in models.values() if row["state"] == state)
    for row in models.values():
        assert (row["state"] == "no_case") == (row["cases"] == [])
        assert (row["state"] == "no_case") == (row["cases_sha256"] is None)


def test_no_shipped_model_fails_or_holds_vacuously(report: dict[str, object]) -> None:
    """Every model that states cases holds them, and at least one case observes activity."""
    totals = report["totals"]
    assert isinstance(totals, dict)
    assert totals["failed"] == 0
    assert totals["held_vacuously"] == 0
    assert totals["invalid"] == 0


def test_the_count_of_models_without_a_case_is_the_pinned_one(report: dict[str, object]) -> None:
    """It falls only in a unit that adds cases and lowers the number here."""
    totals = report["totals"]
    assert isinstance(totals, dict)
    assert totals["no_case"] == MODELS_WITHOUT_A_CASE


def test_the_dendritic_nmda_model_holds_its_cases(report: dict[str, object]) -> None:
    """Coincidence detection as observed counts: neither input alone fires, both do."""
    row = _models(report)["DendriticNMDANeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    observed = {str(case["name"]): case for case in cases}
    assert observed["a soma current of 20 alone stays below threshold"]["observed"] == 0
    assert observed["glutamate alone stays below threshold"]["observed"] == 0
    assert observed["a soma current of 20 with glutamate fires"]["observed"] == 59
    assert observed["a soma current of 40 alone fires"]["observed"] == 190
    assert observed["a soma current of 40 with glutamate fires"]["observed"] == 207
    assert observed["a soma current of 40 with glutamate fires"]["bounds"] == [200, 220]
    contrast = observed["glutamate adds spikes at a soma current of 40"]
    assert (contrast["observed"], contrast["reference_observed"]) == (207, 190)
    assert observed["the lane anchor"]["observed"] == 266
    balance = observed["glutamate alone raises the dendrite to its balance point"]
    assert balance["observed"] == pytest.approx(-59.5204, abs=1e-4)
    assert all(case["held"] for case in cases)
    assert sum(1 for case in cases if case["exercised"]) == 6
    assert balance["bounds"] == [-60.0, -59.0]
    assert contrast["reference_inputs"] == {"glutamate": 0.0, "i_soma": 40.0}
    assert all(case["steps"] == 20000 for case in cases)


def test_the_report_binds_the_cases_the_descriptor_states(report: dict[str, object]) -> None:
    """The digest of the row is the digest of the cases parsed from the shipped descriptor."""
    from sc_neurocore.neurons.model_catalogue import load_descriptor

    descriptor = load_descriptor("DendriticNMDANeuron")
    assert descriptor is not None
    assert len(descriptor.behaviour) == 8
    assert _models(report)["DendriticNMDANeuron"]["cases_sha256"] == cases_digest(
        descriptor.behaviour
    )


def test_the_fitzhugh_nagumo_model_holds_the_cases_of_its_band(report: dict[str, object]) -> None:
    """Silent without a current, 26 and 28 spikes inside the band, one spike above it."""
    row = _models(report)["FitzHughNagumoNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 26, 28, 28, 1]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 0.5},
        {"current": 0.8},
        {"current": 1.0},
        {"current": 2.0},
    ]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, True, True]


def test_the_hindmarsh_rose_model_holds_its_enrolled_and_direction_counts(
    report: dict[str, object],
) -> None:
    """Silent at currents 0 and 2, 26, 40 and 52 upward crossings, then one that tells the direction.

    The first five counts over 2000 steps are those the model's Q16.16 parity test
    enrols for the hand model, the schema and the RTL alike. Each of those windows
    holds as many downward crossings of the threshold as upward ones. At 20.0 x
    crosses upward once and is held above the threshold for the rest of the
    window: 1 event, where a count of downward crossings gives 0.
    """
    row = _models(report)["HindmarshRoseNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 0, 26, 40, 52, 1]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 2.0},
        {"current": 3.0},
        {"current": 4.0},
        {"current": 5.0},
        {"current": 20.0},
    ]
    assert {case["steps"] for case in cases} == {2000}
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, False, True, True, True, True]


def test_the_wilson_hr_model_holds_its_enrolled_and_direction_counts(
    report: dict[str, object],
) -> None:
    """Silent without a current, 44, 46, 49 and 52 upward crossings, then one that tells the direction.

    The first five counts over 5000 steps are those the model's Q16.16 parity
    test enrols for the hand model, the schema and the RTL alike. Each of those
    windows holds as many downward crossings of v_peak as upward ones. Over 20
    steps at 0.1 v has crossed upward once and not yet fallen back: 1 event,
    where a count of downward crossings gives 0.
    """
    row = _models(report)["WilsonHRNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 44, 46, 49, 52, 1]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 0.08},
        {"current": 0.1},
        {"current": 0.14},
        {"current": 0.2},
        {"current": 0.1},
    ]
    assert [case["steps"] for case in cases] == [5000, 5000, 5000, 5000, 5000, 20]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, True, True, True]


def test_the_symmetric_quadratic_if_model_holds_its_enrolled_counts(
    report: dict[str, object],
) -> None:
    """Silent without a current, then 2 to 250 events in 1000 steps as the current grows.

    The counts are those the model's Q16.16 test enrols for the hand model, the
    schemas and the RTL alike.
    """
    row = _models(report)["SCSymmetricQuadraticIFNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 2, 3, 6, 11, 26, 100, 250]
    assert [case["inputs"]["current"] for case in cases] == [
        0.0,
        0.333,
        0.5,
        1.0,
        2.0,
        5.0,
        20.0,
        50.0,
    ]
    assert {case["steps"] for case in cases} == {1000}
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False] + [True] * 7


def test_the_resetting_wilson_model_holds_its_enrolled_counts(report: dict[str, object]) -> None:
    """Silent without a current, then 1 and 4 events in 5000 steps at currents 2 and 10.

    The counts are those the model's Q16.16 test enrols for the hand model, the
    schema and the RTL alike.
    """
    row = _models(report)["SCResettingWilsonHRNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 1, 4]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 2.0},
        {"current": 10.0},
    ]
    assert {case["steps"] for case in cases} == {5000}
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True]


def test_the_integer_qif_model_holds_its_enrolled_count(report: dict[str, object]) -> None:
    """26 events in 400 steps at the integer current 10, as the model's RTL test enrols them.

    The class takes an int32 current and refuses a float, so the case passes the
    integer its descriptor writes, and the report keeps it an integer.
    """
    row = _models(report)["IntegerQIFNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [26]
    assert [case["inputs"] for case in cases] == [{"current": 10}]
    assert [type(case["inputs"]["current"]) for case in cases] == [int]
    assert [case["steps"] for case in cases] == [400]
    assert [(case["held"], case["exercised"]) for case in cases] == [(True, True)]


def test_the_four_state_glif_model_holds_its_enrolled_counts(report: dict[str, object]) -> None:
    """Silent at rest and below threshold, then 23, 54, 86 and 95 events in 1000 steps.

    The counts are those the model's Q16.16 test enrols for the hand model, the
    schema and the RTL alike, at currents 0, 15, 22, 30, 45 and 50.
    """
    row = _models(report)["SCFourStateGLIFNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 0, 23, 54, 86, 95]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 15.0},
        {"current": 22.0},
        {"current": 30.0},
        {"current": 45.0},
        {"current": 50.0},
    ]
    assert {case["steps"] for case in cases} == {1000}
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, False, True, True, True, True]


def test_the_ibarz_tanaka_map_holds_its_enrolled_and_measured_counts(
    report: dict[str, object],
) -> None:
    """Two counts its RTL test enrols, and two measured with the class at robust horizons.

    The map's Q16.16 test enrols no event in 4 steps at a current of -0.5 and 3
    in 30 at 0.2. Those leave the constant branch's terms and the slow variable's
    undecided, so 7 events in 1583 steps at -0.5 and 18 in 1582 at 0 were
    measured with the public class, which the schema matches. At those horizons
    the count holds under perturbations far below rounding, with the nearest
    event of any perturbed run 34 and 17 steps away.
    """
    row = _models(report)["IbarzTanakaMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 3, 7, 18]
    assert [case["inputs"] for case in cases] == [
        {"current": -0.5},
        {"current": 0.2},
        {"current": -0.5},
        {"current": 0.0},
    ]
    assert [case["steps"] for case in cases] == [4, 30, 1583, 1582]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, True]


def test_the_rulkov_map_holds_its_enrolled_and_measured_counts(report: dict[str, object]) -> None:
    """Three counts its RTL test enrols over 30 steps, and three over 1000 measured with the class.

    The map's Q16.16 test enrols 0, 7 and 10 events in 30 steps at currents 0,
    0.5 and 1.5. Those leave the slow variable's terms undecided, so 0, 34 and
    130 events in 1000 steps at the same currents were measured with the public
    class, which the schema matches.
    """
    row = _models(report)["RulkovMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 7, 10, 0, 34, 130]
    assert [case["inputs"] for case in cases] == [{"current": c} for c in (0.0, 0.5, 1.5) * 2]
    assert [case["steps"] for case in cases] == [30, 30, 30, 1000, 1000, 1000]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, False, True, True]


def test_the_courbage_nekorkin_map_holds_its_enrolled_and_measured_cases(
    report: dict[str, object],
) -> None:
    """Four counts its RTL tests enrol, one count measured with the class, and its upper branch.

    The map's Q16.16 and Q32.32 tests enrol no event in 128 steps at a current of
    -0.3, none in 512 and one in 620 without a current, and one in 128 at 0.3.
    Those leave terms of the upper branch, of the regenerative step and of the
    branch points undecided. Measured with the public class, which the schema
    matches: 3 events in 41 steps at 0.1, and x after those 41 steps on the upper
    branch, between 0.5 and the branch's fixed point at y = 0, 0.625.
    """
    row = _models(report)["CourageNekorkinMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 0, 1, 1, 3, 0.555102]
    assert [case["observable"] for case in cases] == ["spikes"] * 5 + ["state:x"]
    assert [case["inputs"] for case in cases] == [
        {"current": c} for c in (-0.3, 0.0, 0.0, 0.3, 0.1, 0.1)
    ]
    assert [case["steps"] for case in cases] == [128, 512, 620, 128, 41, 41]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, False, True, True, True, True]


def test_the_fitzhugh_rinzel_model_holds_its_enrolled_measured_and_direction_counts(
    report: dict[str, object],
) -> None:
    """Three counts its RTL test enrols, one over 39600 steps, and one that tells the direction.

    The model's Q16.16 test enrols 7, 8 and 8 events in 3000 steps at currents
    0.4, 0.5 and 0.6. Over 3000 steps the ultra-slow variable y moves by about a
    hundredth, so the signs of its terms change no count. Over 39600 steps at
    0.4 it shapes the bursts: the public class, which the schema matches, fires
    83 times, and the nearest event of any perturbed run is 1226 steps from the
    horizon. Each of those windows holds as many downward crossings of the
    threshold as upward ones. The model's dynamics test enrols 1 event in 10000
    steps at 2.0: v crosses upward once and stays above the threshold, so a
    count of downward crossings gives 0.
    """
    row = _models(report)["FitzHughRinzelNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [7, 8, 8, 83, 1]
    assert [case["inputs"] for case in cases] == [{"current": c} for c in (0.4, 0.5, 0.6, 0.4, 2.0)]
    assert [case["steps"] for case in cases] == [3000, 3000, 3000, 39600, 10000]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_pernarowski_model_holds_its_enrolled_measured_and_direction_counts(
    report: dict[str, object],
) -> None:
    """Four counts its RTL test enrols, one over 16049 steps, and one that tells the direction.

    The model's Q16.16 test enrols 17 crossings in 5000 steps at each of the
    currents -0.1, 0, 0.1 and 0.2: the oscillator is autonomous, so the current
    shifts its orbit without changing that count. Over 16049 steps at -0.1 the
    public class, which the schema matches, crosses 54 times, and the nearest
    crossing of any perturbed run is 132 steps from the horizon. Each of those
    windows holds as many downward crossings of the threshold as upward ones.
    Over 160 steps at 0.2 v has crossed upward once and not yet fallen back: 1
    event, where a count of downward crossings gives 0.
    """
    row = _models(report)["PernarowskiNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [17, 17, 17, 17, 54, 1]
    assert [case["inputs"] for case in cases] == [
        {"current": c} for c in (-0.1, 0.0, 0.1, 0.2, -0.1, 0.2)
    ]
    assert [case["steps"] for case in cases] == [5000, 5000, 5000, 5000, 16049, 160]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_sc_inclusive_perfect_integrator_holds_its_boundary_and_margin_counts(
    report: dict[str, object],
) -> None:
    """500 events in 1000 steps at current 5 on the inclusive boundary, and 500 in 2000 at 3.

    The source receipt's test enrols the first count for this class: each step
    adds exactly 0.5, so every second candidate equals the threshold 1.0 and an
    inclusive comparison fires there, where a strict one would fire 333 times.
    The count is exact in every arithmetic form of the step, but it sits on the
    boundary by design: a current smaller by 1e-12 gives 333. The second count,
    measured with the class, which the schema matches, keeps every candidate at
    least 0.1 from the threshold, so no perturbed run moves it.
    """
    row = _models(report)["SCInclusivePerfectIntegratorNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [500, 500]
    assert [case["inputs"] for case in cases] == [{"current": 5.0}, {"current": 3.0}]
    assert [case["steps"] for case in cases] == [1000, 2000]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_ermentrout_kopell_map_holds_its_enrolled_and_measured_counts(
    report: dict[str, object],
) -> None:
    """Three counts its RTL test enrols over 2000 steps, and one over 1111 measured with the class.

    The map's Q16.16 test enrols 0, 45 and 64 events in 2000 steps at currents
    -0.5, 0.5 and 1.0 for the hand schema and the RTL. At a current of 0.02 the
    phase takes about 222 steps per turn; the public class, which the schema
    matches, fires 5 times in 1111 steps, and the nearest event of any perturbed
    run is 110 steps from the horizon.
    """
    row = _models(report)["ErmentroutKopellMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 45, 64, 5]
    assert [case["inputs"] for case in cases] == [{"current": c} for c in (-0.5, 0.5, 1.0, 0.02)]
    assert [case["steps"] for case in cases] == [2000, 2000, 2000, 1111]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, True]


def test_the_upward_crossing_rulkov_map_holds_its_window_and_measured_counts(
    report: dict[str, object],
) -> None:
    """Three 30-step windows its RTL test runs, and two counts measured with the class.

    The map's Q16.16 test requires the hand model, the schema and the RTL to give
    the same events over 30 steps at currents 0, 0.5 and 1.5: none, 7 and 10. At
    0.2 the map fires 11 times within its first hundred steps and then falls
    silent, so over 200 steps the public class, which the schema matches, counts
    11, and no event of any perturbed run lies within 104 steps of the horizon.
    Each of those windows holds as many downward crossings as upward ones. Over
    31 steps at 1.5 the window ends on the step of the eleventh upward crossing,
    before x falls back: 11 events, where a count of downward crossings gives 10.
    """
    row = _models(report)["SCUpwardCrossingRulkovMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [0, 7, 10, 11, 11]
    assert [case["inputs"] for case in cases] == [{"current": c} for c in (0.0, 0.5, 1.5, 0.2, 1.5)]
    assert [case["steps"] for case in cases] == [30, 30, 30, 200, 31]
    assert all(case["held"] for case in cases)
    assert [case["exercised"] for case in cases] == [False, True, True, True, True]


def test_the_clipped_logistic_bursting_map_holds_its_enrolled_counts(
    report: dict[str, object],
) -> None:
    """Four 30-step windows its tests enrol, the last on a chaotic trajectory.

    The map's Q16.16 test requires the hand model, the schema and the RTL to give
    the same events over 30 steps at currents 0.5, 1.0 and 2.0: 2, 1 and 1. Its
    sensitive-boundary test enrols the hand model's 7 events over 30 steps at
    0.05, where the RTL fires 8. In double precision the count holds: x stays
    2.1e-3 from the threshold, and the class keeps 7 for every scanned change
    of the current below 5.3e-8.
    """
    row = _models(report)["SCClippedLogisticBurstingMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [2, 1, 1, 7]
    assert [case["inputs"] for case in cases] == [{"current": c} for c in (0.5, 1.0, 2.0, 0.05)]
    assert [case["steps"] for case in cases] == [30, 30, 30, 30]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_adaptive_threshold_map_holds_its_window_and_direction_counts(
    report: dict[str, object],
) -> None:
    """The 32-step window its RTL test runs, and a longer window that tells the crossing's direction.

    The map's Q8.24 test requires the hand model and the RTL to give the same events
    over 32 steps at a current of 0.4: 3, read from the class. That window holds as
    many downward crossings as upward ones. Over 38 steps it ends after the fourth
    upward crossing, before x falls back below the threshold: 4 events, where a
    count of downward crossings gives 3.
    """
    row = _models(report)["SCAdaptiveThresholdMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [3, 4]
    assert [case["inputs"] for case in cases] == [{"current": 0.4}, {"current": 0.4}]
    assert [case["steps"] for case in cases] == [32, 38]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_chialvo_map_states_two_windows_its_q1616_test_runs(report: dict[str, object]) -> None:
    """Two of the 100-step windows its Q16.16 test runs for the hand model and the RTL.

    The map's co-simulation test requires the RTL's event count to equal the hand
    model's over 100 steps at five currents. At 0.01 the map fires 3 times, in a window
    that holds as many downward crossings of the threshold as upward ones. At 1.0 x
    crosses the threshold once and stays above it, so a count of downward crossings
    gives 0 where the map gives 1.
    """
    row = _models(report)["ChialvoMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [3, 1]
    assert [case["inputs"] for case in cases] == [{"current": 0.01}, {"current": 1.0}]
    assert [case["steps"] for case in cases] == [100, 100]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_exponential_if_states_two_windows_its_q3232_test_runs(
    report: dict[str, object],
) -> None:
    """Two of the 1,000-step windows its Q32.32 test runs for the hand model and the RTL.

    The model's co-simulation test requires the hand model, the schema and the RTL to
    give the same event count over 1,000 steps at six currents. At 20.0 the model fires
    twice; the same window's complete RTL event vector is frozen as well. At 100.0 it
    fires 9 times: there the interval between spikes depends on the reset voltage,
    which leaves the count at 20.0 unchanged.
    """
    row = _models(report)["ExpIFNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [2, 9]
    assert [case["inputs"] for case in cases] == [{"current": 20.0}, {"current": 100.0}]
    assert [case["steps"] for case in cases] == [1000, 1000]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_sigma_delta_states_the_window_its_reference_test_runs(
    report: dict[str, object],
) -> None:
    """The 128-step window at 2.0 that its reference test checks against independent equations.

    The test drives 32 steps at 0, which leave the integrator and the reconstruction at
    0, then 128 steps at 2.0, and requires the class to emit the events the equations
    written out in the test give: 42 in that window.
    """
    row = _models(report)["SigmaDeltaNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [42]
    assert [case["inputs"] for case in cases] == [{"current": 2.0}]
    assert [case["steps"] for case in cases] == [128]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_adaptive_threshold_if_states_its_reference_window_and_a_longer_one(
    report: dict[str, object],
) -> None:
    """The window its reference test checks, and a longer one where the threshold relaxes.

    The reference test writes the model's two exact relaxations out itself and requires
    the committed trace's features, among them one event over 160 steps at 20.0. Over
    1,000 steps at the same current the model fires 3 times, the intervals between them
    lengthening as the raised threshold relaxes back towards its rest value.
    """
    row = _models(report)["AdaptiveThresholdIFNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [1, 3]
    assert [case["inputs"] for case in cases] == [{"current": 20.0}, {"current": 20.0}]
    assert [case["steps"] for case in cases] == [160, 1000]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_aihara_map_states_its_q824_window_twice_and_its_level_event(
    report: dict[str, object],
) -> None:
    """The 12-step window its Q8.24 test runs, at 0.0 and at -0.03, and its level test.

    The map's co-simulation test requires the hand model, the schema and the RTL to give
    the same events over 12 steps at 0.0. With the defaults the map then alternates and
    fires on every second step, 6 times. At -0.03, one of the drive values of its
    reference trace, the alternation breaks and the map fires 4 times in the same 12
    steps. In both windows each event follows a step without one, so a count of upward
    crossings of x = 0.5 would give the same numbers. The third case takes the
    parameters of the model's own level test: x stays above 0.5 from the start, and the
    map fires on both steps, where a crossing count gives 0.
    """
    row = _models(report)["AiharaMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [6, 4, 2]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": -0.03},
        {"current": 0.0},
    ]
    assert [case["steps"] for case in cases] == [12, 12, 2]
    assert [case.get("parameters") for case in cases] == [
        None,
        None,
        {"alpha": 0.01, "bias": 0.2, "k": 0.0},
    ]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_nagumo_sato_map_states_its_first_steps_a_drive_and_its_level_event(
    report: dict[str, object],
) -> None:
    """The three steps its model test writes out, a 12-step drive at 0.05, and a level.

    The model test gives y after each of the first three steps at 0.0 as numbers written
    in the test, -0.74, -0.244 and 0.0536, with one event, at the third. At 0.05, a drive
    value of its reference trace and of its Q16.16 co-simulation, the map fires every
    third step, 4 times in 12. In both windows each event follows a step without one, so
    a count of upward crossings of y = 0 would give the same numbers. At 1.0 y stays at
    or above 0.26 from the default start of 0.1, and the map fires on every step, where a
    crossing count gives 0.
    """
    row = _models(report)["NagumoSatoMapNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    assert [case["observed"] for case in cases] == [1, 4, 12]
    assert [case["inputs"] for case in cases] == [
        {"current": 0.0},
        {"current": 0.05},
        {"current": 1.0},
    ]
    assert [case["steps"] for case in cases] == [3, 12, 12]
    assert all(case["held"] for case in cases)
    assert all(case["exercised"] for case in cases)


def test_the_summary_counts_the_states(report: dict[str, object]) -> None:
    """An operator sees how many models state cases and how many hold them."""
    summary = render_summary(report)
    assert "catalogue models: 185" in summary
    assert re.search(rf"^  held\s+{185 - MODELS_WITHOUT_A_CASE}$", summary, flags=re.M)
    assert re.search(rf"^  no_case\s+{MODELS_WITHOUT_A_CASE}$", summary, flags=re.M)
    for note in REPORT_NOTES:
        assert note in summary
    assert any("the author's statement" in note for note in REPORT_NOTES)
    assert any("vector" in note for note in REPORT_NOTES)


def test_write_then_check_and_a_stale_file_fails(tmp_path: Path) -> None:
    """The command writes the report, passes its own check and refuses a changed file."""
    output = tmp_path / "report.json"
    assert main(["--write", "--output", str(output)]) == 0
    assert main(["--check", "--output", str(output)]) == 0
    output.write_text(output.read_text(encoding="utf-8").replace("266", "253"), encoding="utf-8")
    assert main(["--check", "--output", str(output)]) == 1
    assert main(["--check", "--output", str(tmp_path / "absent.json")]) == 1


def test_no_argument_prints_the_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """The bare command prints the summary and writes nothing."""
    assert main([]) == 0
    assert "catalogue models: 185" in capsys.readouterr().out


def test_the_script_runs_as_a_program(monkeypatch: pytest.MonkeyPatch) -> None:
    """The tool runs from its own path, as the generator checks of a unit call it."""
    script = ROOT / "tools" / "model_behaviour_report.py"
    monkeypatch.setattr(sys, "argv", [str(script), "--check"])
    with pytest.raises(SystemExit) as finished:
        runpy.run_path(str(script), run_name="__main__")
    assert finished.value.code == 0
    held = json.loads((ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))["totals"]["held"]
    assert held == 185 - MODELS_WITHOUT_A_CASE


class _Diverges:
    """A model whose state leaves the finite numbers."""

    def __init__(self) -> None:
        self.level = 1.0

    def step(self, current: float) -> int:
        """Square a growing level until it overflows to infinity."""
        self.level = self.level * 1e200 * current
        return 0


class _Vector:
    """A model whose state is a list, not one number."""

    def __init__(self) -> None:
        self.level = [0.0, 0.0]

    def step(self, current: float) -> int:
        """Never fire."""
        return 0


@pytest.mark.parametrize("model", [_Diverges, _Vector])
def test_a_state_that_is_not_one_finite_number_is_invalid(model: type) -> None:
    """A bound cannot be judged against infinity or against a vector."""
    case = _interval("x", {"current": 1.0}, (0.0, 1.0), observable="state:level", steps=3)
    result = evaluate_model(model, (case,))
    assert result["state"] == "invalid"
    assert "single finite number" in result["cases"][0]["reason"]


def test_a_model_without_cases_states_none() -> None:
    """No case, no verdict."""
    assert evaluate_model(_Integrator, ()) == {
        "state": "no_case",
        "cases_sha256": None,
        "cases": [],
    }


def test_the_check_names_a_model_that_fails_or_holds_vacuously() -> None:
    """A report that is committed as computed still fails its check when cases do not hold."""
    failing = evaluate_model(_Integrator, (_interval("fires more", {"current": 1.0}, (20, 30)),))
    vacuous = evaluate_model(_Integrator, (_interval("silent", {"current": 0.0}, (0, 0)),))
    held = evaluate_model(_Integrator, (_interval("fires", {"current": 1.0}, (12, 12)),))
    report = report_of(
        [{"model": "A", **failing}, {"model": "B", **vacuous}, {"model": "C", **held}]
    )
    assert report["totals"] == {
        "held": 1,
        "failed": 1,
        "held_vacuously": 1,
        "invalid": 0,
        "no_case": 0,
    }
    (problem,) = check_problems(report, encode(report))
    assert "'failed': 1" in problem and "'held_vacuously': 1" in problem
    assert (
        check_problems(
            report_of([{"model": "C", **held}]), encode(report_of([{"model": "C", **held}]))
        )
        == []
    )
    assert len(check_problems(report, None)) == 2


_WITH_A_GAIN_OF_AN_EIGHTH = _interval(
    "with a gain of an eighth, a unit current fires on every eighth step",
    {"current": 1.0},
    (12, 12),
    parameters={"gain": 0.125},
)


def test_a_case_runs_a_model_built_with_its_parameters() -> None:
    """The case's parameters build the model; the entry shows them next to the inputs."""
    result = evaluate_model(_Gain, (_WITH_A_GAIN_OF_AN_EIGHTH,))
    assert result["state"] == "held"
    (entry,) = result["cases"]
    assert entry == {
        "name": _WITH_A_GAIN_OF_AN_EIGHTH.name,
        "kind": "interval",
        "observable": "spikes",
        "steps": 100,
        "inputs": {"current": 1.0},
        "parameters": {"gain": 0.125},
        "bounds": [12, 12],
        "observed": 12,
        "held": True,
        "exercised": True,
    }
    assert list(entry)[4:6] == ["inputs", "parameters"]


def test_a_case_can_hold_only_with_its_parameters() -> None:
    """The same statement on the model as built by default fails: the parameters are what holds it."""
    by_default = dataclasses.replace(_WITH_A_GAIN_OF_AN_EIGHTH, parameters=())
    result = evaluate_model(_Gain, (by_default,))
    assert result["state"] == "failed"
    assert result["cases"][0]["observed"] == 25
    assert "parameters" not in result["cases"][0]


def test_the_mihalas_niebur_cases_hold_only_with_panel_m_s_current_jumps(
    report: dict[str, object],
) -> None:
    """Both cases set the two current jumps, which the class leaves at 0, and fail without them.

    With panel M's jumps the first burst brings 8 spikes in 400 steps and both bursts 14 in
    2000; the class as built by default fires regularly, 2 and 10 times.

    Parameters
    ----------
    report : dict[str, object]
        The report as the tool computes it.
    """
    from sc_neurocore.neurons.model_catalogue import load_descriptor
    from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron

    row = _models(report)["MihalasNieburNeuron"]
    assert row["state"] == "held"
    cases = row["cases"]
    assert isinstance(cases, list)
    jumps = {"current_jump_1": 0.01, "current_jump_2": -0.0006}
    assert [(case["parameters"], case["observed"]) for case in cases] == [(jumps, 8), (jumps, 14)]
    descriptor = load_descriptor("MihalasNieburNeuron")
    assert descriptor is not None
    by_default = tuple(dataclasses.replace(case, parameters=()) for case in descriptor.behaviour)
    result = evaluate_model(MihalasNieburNeuron, by_default)
    assert result["state"] == "failed"
    assert [case["observed"] for case in result["cases"]] == [2, 10]


def test_whether_a_case_was_exercised_is_judged_on_the_model_with_its_parameters() -> None:
    """At a gain of zero nothing fires, though the model as built by default fires at this drive."""
    silent = _interval(
        "with no gain a unit current fires nothing",
        {"current": 1.0},
        (0, 0),
        parameters={"gain": 0.0},
    )
    result = evaluate_model(_Gain, (silent,))
    assert result["state"] == "held_vacuously"
    assert (result["cases"][0]["observed"], result["cases"][0]["exercised"]) == (0, False)


def test_a_contrast_runs_both_drives_with_the_case_s_parameters() -> None:
    """At a gain of an eighth a unit current fires 12 times and half of it 6 times.

    Built by default, the half current would fire 12 times and the margin would fail.
    """
    contrast = BehaviourCase(
        name="with a gain of an eighth, a unit current fires 6 more than half of it",
        kind="contrast",
        steps=100,
        observable="spikes",
        inputs=(("current", 1.0),),
        reference_inputs=(("current", 0.5),),
        margin=6.0,
        parameters=(("gain", 0.125),),
    )
    result = evaluate_model(_Gain, (contrast,))
    assert result["state"] == "held"
    (entry,) = result["cases"]
    assert (entry["observed"], entry["reference_observed"]) == (12, 6)
    assert entry["parameters"] == {"gain": 0.125}


@pytest.mark.parametrize(
    ("parameters", "reason"),
    [
        ({"gain": -1.0}, "the model cannot be built with the parameters ['gain']: ValueError"),
        ({"bias": 1.0}, "the model cannot be built with the parameters ['bias']: TypeError"),
    ],
)
def test_a_model_that_refuses_a_case_s_parameters_makes_it_invalid(
    parameters: dict[str, float], reason: str
) -> None:
    """A value the model's validation refuses, or a name its constructor does not take, is reported.

    Parameters
    ----------
    parameters : dict
        What the case sets.
    reason : str
        The reason the report must give.
    """
    case = _interval("refused", {"current": 1.0}, (12, 12), parameters=parameters)
    result = evaluate_model(_Gain, (case,))
    assert result["state"] == "invalid"
    assert result["cases"][0]["reason"] == reason
    assert result["cases"][0]["parameters"] == parameters
