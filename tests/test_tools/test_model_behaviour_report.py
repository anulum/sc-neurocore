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
MODELS_WITHOUT_A_CASE = 184


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


def _interval(
    name: str,
    inputs: dict[str, float],
    bounds: tuple[float, float],
    *,
    observable: str = "spikes",
    steps: int = 100,
) -> BehaviourCase:
    return BehaviourCase(
        name=name,
        kind="interval",
        steps=steps,
        observable=observable,
        inputs=tuple(sorted(inputs.items())),
        bounds=bounds,
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
    assert "ValueError" in result["cases"][0]["reason"]


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
        "glutamate",
    ),
    "a required input left out": (_TwoInputs, _interval("x", {"drive": 1.0}, (1, 100)), "gate"),
    "a state variable the model does not hold": (
        _Integrator,
        _interval("x", {"current": 1.0}, (0.0, 1.0), observable="state:calcium"),
        "calcium",
    ),
    "a step that refuses the drive": (
        _Refuses,
        _interval("x", {"current": -1.0}, (0, 0)),
        "ValueError",
    ),
    "a spike count of a step that returns a rate": (
        _Rate,
        _interval("x", {"current": 1.0}, (1, 100)),
        "0 or 1",
    ),
    "a step that needs a whole number": (
        _IndexedByItsInput,
        _interval("x", {"channel": 1.0}, (1, 100)),
        "TypeError",
    ),
    "a model that cannot be built without arguments": (
        _NeedsAnArgument,
        _interval("x", {"current": 1.0}, (0, 0)),
        "built",
    ),
}


@pytest.mark.parametrize("reason", sorted(_INVALID))
def test_a_case_that_cannot_be_run_is_invalid_and_says_why(reason: str) -> None:
    """Nothing is observed and nothing is held; the reason names the fault."""
    model, case, fragment = _INVALID[reason]
    result = evaluate_model(model, (case,))
    assert result["state"] == "invalid"
    (reported,) = result["cases"]
    assert reported["held"] is False
    assert "observed" not in reported
    assert fragment in reported["reason"]


def test_a_state_variable_of_a_model_that_returns_a_rate_can_be_bounded() -> None:
    """What the step returns is read for a spike count only; a state case does not need it."""
    case = _interval(
        "the rate is half the drive", {"current": 3.0}, (1.5, 1.5), observable="state:rate"
    )
    result = evaluate_model(_Rate, (case,))
    assert result["state"] == "held"
    assert result["cases"][0]["observed"] == 1.5
    assert result["cases"][0]["bounds"] == [1.5, 1.5]


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


def test_the_stated_cases_of_a_model_are_bound_by_one_digest() -> None:
    """The digest is of the cases as stated, in order; any change of a statement changes it."""
    first = _interval("silent without drive", {"current": 0.0}, (0, 0))
    second = _interval("fires", {"current": 1.0}, (12, 12))
    result = evaluate_model(_Integrator, (first, second))
    canonical = json.dumps(
        [dataclasses.asdict(first), dataclasses.asdict(second)],
        sort_keys=True,
        separators=(",", ":"),
    )
    assert result["cases_sha256"] == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert result["cases_sha256"] == cases_digest((first, second))
    assert cases_digest((second, first)) != result["cases_sha256"]
    for changed in (
        dataclasses.replace(second, bounds=(12.0, 13.0)),
        dataclasses.replace(second, steps=101),
        dataclasses.replace(second, inputs=(("current", 1.5),)),
        dataclasses.replace(second, name="fires twelve times"),
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


def test_the_summary_counts_the_states(report: dict[str, object]) -> None:
    """An operator sees how many models state cases and how many hold them."""
    summary = render_summary(report)
    assert "catalogue models: 185" in summary
    assert re.search(r"^  held\s+1$", summary, flags=re.M)
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
    assert json.loads((ROOT / DEFAULT_OUTPUT).read_text(encoding="utf-8"))["totals"]["held"] == 1


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
