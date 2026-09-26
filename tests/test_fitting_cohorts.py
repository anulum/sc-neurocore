# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Scientific cohort execution, replay and leakage checks

"""Public cohort protocols exercise real DSL neurons under shared samples."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Literal

import pytest

from sc_neurocore.fitting.cohort import (
    CohortMetric,
    CohortModel,
    CohortSample,
    ExperimentCohort,
    SweepDomain,
    cohort_from_dict,
    cohort_sha256,
)
from sc_neurocore.fitting.cohort_run import replay_cohort, run_cohort
from sc_neurocore.fitting.constraints import ParameterConstraint
from sc_neurocore.fitting.problem import simulate
from sc_neurocore.fitting.pareto import measured_pareto
from sc_neurocore.neurons.universal_dsl import load_schema


def cohort() -> ExperimentCohort:
    """Create a LIF protocol with distinct held-out acquisitions and explicit noise."""
    schema = load_schema("lif")
    samples = []
    entries: tuple[tuple[str, float, Literal["train", "holdout"]], ...] = (
        ("train", 1.0, "train"),
        ("holdout", 2.0, "holdout"),
    )
    for name, level, split in entries:
        current = (level,) * 12
        noise = tuple(0.05 if index % 2 else -0.05 for index in range(12))
        observed = simulate(
            schema, "v", {}, tuple(a + b for a, b in zip(current, noise, strict=True))
        )
        assert observed is not None
        samples.append(
            CohortSample(name, name, split, current, noise, {"v": tuple(observed)}, (0,) * 12)
        )
    models = (
        CohortModel(
            "voltage",
            schema,
            (SweepDomain("R", (1.0, 2.0)),),
            CohortMetric("trace_rmse", "v", "mV"),
        ),
        CohortModel(
            "events",
            schema,
            (SweepDomain("R", (1.0, 2.0)),),
            CohortMetric("event_disagreement", "", "fraction"),
        ),
    )
    return ExperimentCohort(
        "shared step",
        tuple(samples),
        models,
        1.0,
        "ms",
        "nA",
        7,
        "explicit alternating +/-0.05 nA samples; deterministic test stimulus",
    )


def test_full_cohort_round_trip_and_shared_noise() -> None:
    """All models use the same inputs and export every trial for exact replay."""
    study = cohort()
    result = run_cohort(cohort_from_dict(study.to_public_dict()))
    assert len(result["trials"]) == 4
    assert result["trials"][0]["samples"][0]["value"] == 0
    assert result["trials"][1]["samples"][0]["value"] > 0
    assert result["selection"][0]["trial_sha256"] == result["trials"][0]["trial_sha256"]
    assert replay_cohort(result)["reproduced"] is True
    assert result["provenance"]["sample_sha256"]["train"] == cohort_sha256(
        study.samples[0].to_public_dict()
    )


def test_heldout_values_and_failures_never_select_sweep_parameters() -> None:
    """Changing held-out observations cannot influence training-only model selection."""
    study = cohort()
    original = run_cohort(study)
    changed = replace(study.samples[1], observations={"v": (1e308,) * 12})
    again = run_cohort(replace(study, samples=(study.samples[0], changed)))
    assert again["trials"][0]["status"] == "failed"
    assert (
        original["selection"][0]["training_metric"] == again["selection"][0]["training_metric"] == 0
    )
    selected = next(
        t for t in again["trials"] if t["trial_sha256"] == again["selection"][0]["trial_sha256"]
    )
    assert selected["parameters"]["R"] == 1


def test_constraints_and_all_failed_candidates_stay_visible() -> None:
    """The full declared grid is retained even if a constraint rejects every member."""
    study = cohort()
    constraint = ParameterConstraint("R below C", {"R": 1.0, "C": -1.0}, -10.0, 0.1)
    model = replace(study.models[0], constraints=(constraint,))
    result = run_cohort(replace(study, models=(model,)))
    assert [t["status"] for t in result["trials"]] == ["completed", "constraint_rejected"]
    assert result["trials"][1]["rejected_constraints"] == [constraint.name]
    all_rejected = run_cohort(
        replace(study, models=(replace(model, domains=(SweepDomain("R", (2.0, 3.0)),)),))
    )
    assert all_rejected["selection"][0]["trial_sha256"] is None


def test_spike_count_is_a_model_specific_metric() -> None:
    """A declared count error scores events from real neuron steps."""
    study = cohort()
    model = replace(study.models[1], metric=CohortMetric("spike_count_error", "", "events"))
    samples = tuple(replace(s, spikes=(1,) * 12) for s in study.samples)
    result = run_cohort(replace(study, models=(model,), samples=samples))
    assert result["trials"][0]["samples"][0]["value"] == 12


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"dt": 2.0}, "declared timebase"),
        ({"dt": float("nan")}, "positive dt"),
        ({"seed": -1}, "nonnegative seed"),
        ({"input_unit": ""}, "input units"),
        ({"noise_provenance": ""}, "noise provenance"),
        ({"models": ()}, "named models"),
    ],
)
def test_bad_cohort_contract_is_refused(patch: dict[str, Any], message: str) -> None:
    """Invalid scientific contracts fail before any trial runs."""
    with pytest.raises(ValueError, match=message):
        replace(cohort(), **patch)


def test_split_leakage_and_names_are_refused() -> None:
    """Acquisition group, content and naming custody remain explicit."""
    study = cohort()
    with pytest.raises(ValueError, match="acquisition groups"):
        replace(study, samples=(study.samples[0], replace(study.samples[1], group="train")))
    with pytest.raises(ValueError, match="data cannot cross"):
        replace(
            study,
            samples=(
                study.samples[0],
                replace(study.samples[0], name="other", group="other", split="holdout"),
            ),
        )
    with pytest.raises(ValueError, match="both train and holdout"):
        replace(study, samples=(study.samples[0],))
    with pytest.raises(ValueError, match="names must be unique"):
        replace(study, models=(study.models[0], study.models[0]))


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        (("R", (), "real"), "finite values"),
        (("R", (float("nan"),), "real"), "finite values"),
        (("R", (1.0, 1.0), "real"), "unique"),
        (("R", (1.5,), "integer"), "fractional"),
        (("R", (1.0,), "log"), "real/integer"),
    ],
)
def test_typed_domains_refuse_bad_values(domain: tuple[Any, ...], message: str) -> None:
    """Domain typing prevents silent rounding and duplicate trials."""
    with pytest.raises(ValueError, match=message):
        SweepDomain(*domain)


def test_integer_domains_and_budget_admission() -> None:
    """Integral domains are admitted; explosive Cartesian grids are refused intact."""
    study = cohort()
    model = replace(study.models[0], domains=(SweepDomain("R", (1.0, 2.0), "integer"),))
    assert len(run_cohort(replace(study, models=(model,)))["trials"]) == 2
    with pytest.raises(ValueError, match="not shortened"):
        replace(
            study,
            models=(
                replace(model, domains=(SweepDomain("R", tuple(float(i) for i in range(4097))),)),
            ),
        )


def receipts(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Build receipt documents for validator tests, without claiming hardware acquisition."""
    rows = []
    contract = {
        "target": "document-validation-test",
        "device_revision": "fixture",
        "harness_sha256": "a" * 64,
        "workload_sha256": result["provenance"]["cohort_sha256"],
        "warmup": "10 runs excluded",
        "transport": "included",
        "repeats": 3,
        "aggregation": "median",
        "resource_unit": "LUT",
        "instrument": "receipt syntax fixture",
        "calibration_sha256": "c" * 64,
    }
    for index, trial in enumerate(result["trials"][:2]):
        row = {
            "schema_version": "sc-neurocore.measurement.v1",
            "source_kind": "physical",
            "cohort_sha256": result["provenance"]["cohort_sha256"],
            "trial_sha256": trial["trial_sha256"],
            "contract": contract,
            "latency_ms": 1.0 + index,
            "resources": 10.0 + index,
            "energy_j": 0.1 + index,
        }
        row["receipt_sha256"] = cohort_sha256(row)
        rows.append(row)
    return rows


def test_measurement_frontier_requires_comparable_receipt_contracts() -> None:
    """Receipt syntax validation cannot turn missing or incomparable data into a frontier."""
    result = run_cohort(cohort())
    documents = receipts(result)
    report = measured_pareto(result, documents)
    assert report["comparable"] is True
    assert [row["nondominated"] for row in report["rows"]] == [True, False]
    assert "not independently verified" in report["custody"]
    assert measured_pareto(result, [])["comparable"] is False
    documents[1]["contract"] = {**documents[1]["contract"], "resource_unit": "bytes"}
    documents[1]["receipt_sha256"] = cohort_sha256(
        {k: v for k, v in documents[1].items() if k != "receipt_sha256"}
    )
    assert "must match" in measured_pareto(result, documents)["reason"]


def test_receipt_custody_digests_and_synthetic_sources_are_refused() -> None:
    """A receipt edit, unrelated result or synthetic acquisition yields no claim."""
    result = run_cohort(cohort())
    documents = receipts(result)
    edited = copy.deepcopy(documents)
    edited[0]["energy_j"] = 0
    assert measured_pareto(result, edited)["comparable"] is False
    synthetic = copy.deepcopy(documents)
    synthetic[0]["source_kind"] = "synthetic"
    synthetic[0]["receipt_sha256"] = cohort_sha256(
        {k: v for k, v in synthetic[0].items() if k != "receipt_sha256"}
    )
    assert measured_pareto(result, synthetic)["comparable"] is False
    assert measured_pareto({**result, "result_sha256": "0" * 64}, documents)["comparable"] is False


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"name": ""}, "samples need names"),
        ({"group": ""}, "samples need names"),
        ({"split": "bad"}, "splits"),
        ({"noise": (0.0,)}, "equal lengths"),
        ({"current": (float("inf"),) * 12}, "must be finite"),
        ({"spikes": (2,) * 12}, "binary"),
        ({"observations": {"v": (0.0,)}}, "observations need"),
        ({"current": (1e308,) * 12, "noise": (1e308,) * 12}, "effective current"),
    ],
)
def test_sample_protocol_validation(patch: dict[str, Any], message: str) -> None:
    """Malformed stimuli never reach a neuron step."""
    with pytest.raises(ValueError, match=message):
        replace(cohort().samples[0], **patch)


@pytest.mark.parametrize(
    "metric",
    [
        ("bad", "", ""),
        ("trace_rmse", "", "mV"),
        ("trace_rmse", "v", ""),
        ("event_disagreement", "v", "fraction"),
        ("spike_count_error", "", "fraction"),
    ],
)
def test_metric_contract_validation(metric: tuple[Any, ...]) -> None:
    """Unlike event and voltage measurements cannot be labelled interchangeably."""
    with pytest.raises(ValueError):
        CohortMetric(*metric)


def test_model_parameter_units_observations_and_versions_are_checked() -> None:
    """Models must share declared semantic references and a complete recording contract."""
    study = cohort()
    model = study.models[0]
    invalid: list[dict[str, Any]] = [
        {"name": ""},
        {"fixed": {"R": 1.0}},
        {"fixed": {"unknown": 1.0}},
        {"fixed": {"C": float("nan")}},
        {"domains": (model.domains[0], model.domains[0])},
        {"metric": CohortMetric("trace_rmse", "v", "V")},
    ]
    for patch in invalid:
        with pytest.raises(ValueError):
            replace(model, **patch)
    c = ParameterConstraint("same", {"R": 1.0}, 0.0, 2.0)
    with pytest.raises(ValueError):
        replace(model, constraints=(c, c))
    with pytest.raises(ValueError):
        replace(model, constraints=(ParameterConstraint("unknown", {"x": 1.0}, 0.0, 2.0),))
    with pytest.raises(ValueError, match="observations in every"):
        replace(study, samples=(replace(study.samples[0], observations={}), study.samples[1]))
    with pytest.raises(ValueError, match="unsupported cohort"):
        cohort_from_dict({"schema_version": "another"})


def test_diverging_models_fail_in_full_cohort_and_have_no_training_selection() -> None:
    """A real explosive ODE yields visible failed trials, never a truncated trace."""
    study = cohort()
    schema = {
        "metadata": {"schema_version": 2, "name": "explosive"},
        "state": {"v": 1.0},
        "parameters": {"gain": 1.0},
        "integration": {"dt": 1.0, "method": "euler"},
        "dynamics": {"v": "gain*v"},
        "profile": {"time_unit": "ms", "units": {"v": "mV"}},
    }
    model = CohortModel(
        "unstable", schema, (SweepDomain("gain", (1e300,)),), CohortMetric("trace_rmse", "v", "mV")
    )
    result = run_cohort(replace(study, models=(model,)))
    assert result["trials"][0]["status"] == "failed"
    assert result["selection"][0]["trial_sha256"] is None


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"cohort_sha256": "0" * 64}, "different cohort"),
        ({"trial_sha256": "missing"}, "missing"),
        ({"latency_ms": float("inf")}, "finite nonnegative"),
        ({"energy_j": -1.0}, "finite nonnegative"),
        ({"resources": True}, "finite nonnegative"),
    ],
)
def test_measurement_receipts_refuse_bad_binding_and_axes(
    change: dict[str, Any], message: str
) -> None:
    """Receipt digests cannot legitimise invalid measurement axes or unrelated trials."""
    result = run_cohort(cohort())
    documents = receipts(result)
    documents[0].update(change)
    documents[0]["receipt_sha256"] = cohort_sha256(
        {k: v for k, v in documents[0].items() if k != "receipt_sha256"}
    )
    report = measured_pareto(result, documents)
    assert not report["comparable"]
    assert message in report["reason"]


def test_measurement_contracts_duplicates_and_failed_trials_are_refused() -> None:
    """An incomplete acquisition contract or failed scientific trial yields no frontier."""
    result = run_cohort(cohort())
    assert "duplicate" in measured_pareto(result, [receipts(result)[0]] * 2)["reason"]
    for patch in (
        {"repeats": 0},
        {"repeats": True},
        {"instrument": ""},
        {"calibration_sha256": "bad"},
    ):
        documents = receipts(result)
        documents[0]["contract"] = {**documents[0]["contract"], **patch}
        documents[0]["receipt_sha256"] = cohort_sha256(
            {k: v for k, v in documents[0].items() if k != "receipt_sha256"}
        )
        assert not measured_pareto(result, documents)["comparable"]
    failed = copy.deepcopy(result)
    failed["trials"][0]["status"] = "failed"
    failed["result_sha256"] = cohort_sha256(
        {k: v for k, v in failed.items() if k != "result_sha256"}
    )
    assert "failed trials" in measured_pareto(failed, receipts(failed))["reason"]


def test_trace_metrics_do_not_require_invented_spike_observations() -> None:
    """Trace-only cohorts may omit spikes, but event comparisons require them."""
    study = cohort()
    samples = tuple(replace(s, spikes=()) for s in study.samples)
    result = run_cohort(replace(study, models=(study.models[0],), samples=samples))
    assert result["trials"][0]["status"] == "completed"
    with pytest.raises(ValueError, match="event metrics need recorded"):
        replace(study, samples=samples)


@pytest.mark.parametrize(
    ("field", "value"), [("seed", 0.5), ("seed", True), ("seed", 2**53), ("name", {}), ("dt", True)]
)
def test_document_types_cannot_be_silently_coerced(field: str, value: Any) -> None:
    """JSON custody rejects rounded seeds and stringified identities."""
    document = cohort().to_public_dict()
    document[field] = value
    with pytest.raises(ValueError):
        cohort_from_dict(document)


def test_unknown_fields_boolean_values_and_step_budgets_are_refused() -> None:
    """Unknown definitions and hidden over-budget work are never ignored or shortened."""
    document = cohort().to_public_dict()
    document["unrecognised_noise"] = 1
    with pytest.raises(ValueError, match="missing or unknown"):
        cohort_from_dict(document)
    document = cohort().to_public_dict()
    document["samples"][0]["noise"][0] = True
    with pytest.raises(ValueError, match="JSON numbers"):
        cohort_from_dict(document)
    study = cohort()
    samples = tuple(
        replace(
            s,
            current=(s.current[0],) * 1300,
            noise=(0.0,) * 1300,
            observations={"v": (0.0,) * 1300},
            spikes=(0,) * 1300,
        )
        for s in study.samples
    )
    model = replace(
        study.models[0], domains=(SweepDomain("R", tuple(float(i) for i in range(2000))),)
    )
    with pytest.raises(ValueError, match="model-step budget"):
        replace(study, models=(model,), samples=samples)


def test_measurement_workload_must_bind_the_whole_cohort() -> None:
    """A valid receipt hash with a different workload cannot enter a common frontier."""
    result = run_cohort(cohort())
    documents = receipts(result)
    documents[0]["contract"] = {**documents[0]["contract"], "workload_sha256": "d" * 64}
    documents[0]["receipt_sha256"] = cohort_sha256(
        {k: v for k, v in documents[0].items() if k != "receipt_sha256"}
    )
    assert "complete shared-sample" in measured_pareto(result, documents)["reason"]


def test_cohort_digest_survives_browser_numeric_roundtrip() -> None:
    """Equivalent int/float JSON values retain scientific and receipt custody."""
    result = run_cohort(cohort())

    def browser_numbers(value: Any) -> Any:
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, dict):
            return {key: browser_numbers(item) for key, item in value.items()}
        if isinstance(value, list):
            return [browser_numbers(item) for item in value]
        return value

    transported = browser_numbers(result)
    assert replay_cohort(transported)["reproduced"] is True
    report = measured_pareto(transported, receipts(transported))
    assert report["comparable"] is True


def test_cli_example_exports_complete_protocol_and_result_without_overwriting(
    tmp_path: Any,
) -> None:
    """The released example is a real entry point for creating Studio imports."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    protocol = tmp_path / "cohort.json"
    result_file = tmp_path / "result.json"
    for destination, run in ((protocol, False), (result_file, True)):
        command = [
            sys.executable,
            str(root / "examples/studio_cohort.py"),
            "--output",
            str(destination),
        ]
        if run:
            command.append("--run")
        subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    document = json.loads(protocol.read_text())
    result = json.loads(result_file.read_text())
    assert cohort_from_dict(document).trial_count == 6
    assert result["cohort"] == document
    assert replay_cohort(result)["reproduced"] is True
    refused = subprocess.run(
        [sys.executable, str(root / "examples/studio_cohort.py"), "--output", str(protocol)],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0
    assert json.loads(protocol.read_text()) == document


def test_unsafe_integer_samples_and_nonmapping_document_rows_are_refused() -> None:
    """Scientific documents never silently round integer samples during JSON transport."""
    document = cohort().to_public_dict()
    document["samples"][0]["current"][0] = 2**53 + 1
    with pytest.raises(ValueError, match="round-trip exactly"):
        cohort_from_dict(document)
    document = cohort().to_public_dict()
    document["models"][0]["domains"] = [42]
    with pytest.raises(ValueError, match="missing or unknown"):
        cohort_from_dict(document)


def test_progress_callbacks_cannot_mutate_the_executed_cohort_snapshot() -> None:
    """External nested schema edits cannot change later trials or their exported protocol."""
    study = cohort()
    schema = study.models[0].schema

    def edit_original(event: dict[str, Any]) -> None:
        schema["parameters"]["C"] = 2.0

    result = run_cohort(study, progress=edit_original)
    assert result["cohort"]["models"][0]["schema"]["parameters"]["C"] == 1.0
    assert result["trials"][0]["samples"][0]["value"] == 0.0
    assert replay_cohort(result)["reproduced"] is True


def test_constraint_documents_and_seed_custody_refuse_unknown_or_unrepresentable_fields() -> None:
    """Constraints preserve their full definition and seed custody survives browser transport."""
    document = cohort().to_public_dict()
    document["models"][0]["constraints"] = [
        {
            "name": "bound",
            "coefficients": {"R": 1.0},
            "low": 0.0,
            "high": 2.0,
            "unrecognised": "condition",
        }
    ]
    with pytest.raises(ValueError, match="constraint documents"):
        cohort_from_dict(document)
    document = cohort().to_public_dict()
    document["seed"] = 2**53
    with pytest.raises(ValueError, match="safe nonnegative seed"):
        cohort_from_dict(document)


@pytest.mark.parametrize(
    "patch",
    [
        {"name": 1},
        {"coefficients": []},
        {"low": True},
        {"high": "2"},
        {"coefficients": {"R": False}},
    ],
)
def test_constraints_never_coerce_ambiguous_json_fields(patch: dict[str, Any]) -> None:
    """Boolean coefficients and string bounds cannot silently change a constraint."""
    document = cohort().to_public_dict()
    document["models"][0]["constraints"] = [
        {"name": "bound", "coefficients": {"R": 1.0}, "low": 0.0, "high": 2.0, **patch}
    ]
    with pytest.raises(ValueError):
        cohort_from_dict(document)
