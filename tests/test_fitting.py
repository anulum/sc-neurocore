# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parameter fitting: recovery, held-out error and identifiability

"""Fits recover known parameters, and say so when the data cannot.

The benchmark is the bundled leaky integrate-and-fire schema driven below
threshold by steps of different sizes, recorded with seeded Gaussian noise. Its
resting potential, membrane time constant and resistance are identifiable from
such data; resistance and capacitance together are not, because the dynamics
see only their ratio, and the fit must say that instead of reporting errors.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest

from sc_neurocore.fitting import (
    FIT_SCHEMA_VERSION,
    UNCERTAINTY_METHOD,
    FitProblem,
    ParameterDomain,
    Recording,
    fit_parameters,
    problem_from_dict,
    replay_fit,
    simulate,
)
from sc_neurocore.fitting.fit import _identifiability, _Objective
from sc_neurocore.neurons.universal_dsl import load_schema

TRUTH = {"v_rest": -65.0, "tau_m": 10.0, "R": 1.0, "C": 1.0}


def _schema() -> dict[str, Any]:
    return load_schema("lif")


def _recording(name: str, level: float, rng: np.random.Generator, samples: int = 150) -> Recording:
    current = [0.0] * 20 + [level] * (samples - 20)
    trace = simulate(_schema(), "v", TRUTH, current)
    assert trace is not None
    noisy = trace + rng.normal(0.0, 0.2, samples)
    return Recording(name, tuple(current), tuple(float(value) for value in noisy))


@pytest.fixture(scope="module")
def cohort() -> tuple[tuple[Recording, ...], tuple[Recording, ...]]:
    rng = np.random.default_rng(7)
    train = (_recording("step +5", 5.0, rng), _recording("step -4", -4.0, rng))
    holdout = (_recording("step +8", 8.0, rng),)
    return train, holdout


def _problem(cohort: Any, **overrides: Any) -> FitProblem:
    train, holdout = cohort
    fields: dict[str, Any] = {
        "schema": _schema(),
        "observable": "v",
        "domains": (
            ParameterDomain("v_rest", -80.0, -50.0),
            ParameterDomain("tau_m", 1.0, 50.0, "log"),
            ParameterDomain("R", 0.1, 10.0, "log"),
        ),
        "fixed": {"C": 1.0},
        "train": train,
        "holdout": holdout,
        "seed": 3,
    }
    fields.update(overrides)
    return FitProblem(**fields)


class TestFit:
    @pytest.fixture(scope="class")
    def result(self, cohort: Any) -> dict[str, Any]:
        return fit_parameters(_problem(cohort))

    def test_known_parameters_are_recovered_within_their_stated_uncertainty(
        self, result: dict[str, Any]
    ) -> None:
        assert result["identifiability"]["identifiable"] is True
        errors = result["uncertainty"]["standard_errors"]
        for name in ("v_rest", "tau_m", "R"):
            assert abs(result["fitted"][name] - TRUTH[name]) <= 4.0 * errors[name], name
            assert errors[name] > 0
        assert result["uncertainty"]["method"] == UNCERTAINTY_METHOD
        assert result["uncertainty"]["correlation"]["R"]["R"] == pytest.approx(1.0)

    def test_held_out_error_is_measured_on_data_the_fit_never_saw(
        self, result: dict[str, Any]
    ) -> None:
        (held,) = result["holdout"]
        assert held["recording"] == "step +8"
        assert held["diverged"] is False
        # The noise is 0.2 mV: a model that generalises errs by about that much.
        assert 0.1 < held["rmse"] < 0.3
        assert [row["recording"] for row in result["training"]] == ["step +5", "step -4"]

    def test_the_optimiser_history_and_trial_counts_are_kept(self, result: dict[str, Any]) -> None:
        optimiser = result["optimiser"]
        assert optimiser["evaluations"] > 100
        assert optimiser["failed_trials"] == 0
        losses = [entry["loss"] for entry in optimiser["history"]]
        assert losses and all(later <= earlier for earlier, later in zip(losses, losses[1:]))
        assert result["training_loss"] <= losses[-1]

    def test_the_result_carries_its_problem_and_provenance_under_one_digest(
        self, result: dict[str, Any], cohort: Any
    ) -> None:
        assert result["schema_version"] == FIT_SCHEMA_VERSION
        assert problem_from_dict(result["problem"]) == _problem(cohort)
        assert result["provenance"]["seed"] == 3
        assert len(result["result_sha256"]) == 64


def test_parameters_seen_only_as_a_ratio_are_reported_unidentifiable(cohort: Any) -> None:
    result = fit_parameters(
        _problem(
            cohort,
            domains=(
                ParameterDomain("R", 0.1, 10.0, "log"),
                ParameterDomain("C", 0.1, 10.0, "log"),
            ),
            fixed={"v_rest": -65.0, "tau_m": 10.0},
        )
    )
    diagnosis = result["identifiability"]
    assert diagnosis["identifiable"] is False
    (direction,) = diagnosis["unconstrained_directions"]
    # log R and log C move together: the ratio R / C is all the data constrain.
    components = direction["direction"]
    assert set(components) == {"R", "C"}
    assert math.isclose(abs(components["R"]), abs(components["C"]), rel_tol=1e-3)
    assert result["uncertainty"]["standard_errors"] is None
    assert result["uncertainty"]["correlation"] is None
    assert result["fitted"]["R"] / result["fitted"]["C"] == pytest.approx(1.0, rel=0.02)


def test_a_parameter_the_data_never_touch_is_unconstrained(cohort: Any) -> None:
    result = fit_parameters(
        _problem(
            cohort,
            domains=(ParameterDomain("C", 0.1, 10.0, "log"),),
            fixed={"v_rest": -65.0, "tau_m": 10.0, "R": 0.0},
        ),
        generations=2,
        population=4,
    )
    diagnosis = result["identifiability"]
    assert diagnosis["identifiable"] is False
    assert diagnosis["condition_number"] is None
    assert diagnosis["unconstrained_directions"][0]["relative_eigenvalue"] == 0.0


def _explosive() -> dict[str, Any]:
    """A model whose state overflows when its gain passes a threshold within 150 steps."""
    return {
        "metadata": {"schema_version": 2, "name": "Explosive"},
        "state": {"v": 1.0},
        "parameters": {"k": 1.0},
        "integration": {"dt": 1.0, "method": "euler"},
        "dynamics": {"v": "k * v"},
    }


def _explosive_problem(**overrides: Any) -> FitProblem:
    current = tuple([0.0] * 150)
    fields: dict[str, Any] = {
        "schema": _explosive(),
        "observable": "v",
        "domains": (ParameterDomain("k", 0.0, 400.0),),
        "train": (Recording("flat", current, tuple([1.0] * 150)),),
        "holdout": (Recording("flat again", current, tuple([2.0] * 150)),),
        "seed": 1,
    }
    fields.update(overrides)
    return FitProblem(**fields)


def test_trials_whose_state_overflows_are_counted_as_failed() -> None:
    """Above a gain of about ten the squared residual overflows; above about 111 the state does."""
    result = fit_parameters(
        _explosive_problem(domains=(ParameterDomain("k", 0.0, 20.0),)), generations=8, population=8
    )
    assert 0 < result["optimiser"]["failed_trials"] < result["optimiser"]["evaluations"]
    assert result["fitted"]["k"] < 1e-3
    assert result["training_loss"] < 1e-6


def test_a_fit_that_finds_no_finite_trial_does_not_claim_convergence() -> None:
    result = fit_parameters(
        _explosive_problem(domains=(ParameterDomain("k", 150.0, 400.0),)),
        generations=2,
        population=4,
    )
    assert result["optimiser"]["failed_trials"] == result["optimiser"]["evaluations"]
    assert result["optimiser"]["converged"] is False
    assert result["identifiability"]["identifiable"] is False


def test_near_a_divergence_the_diagnosis_says_so_instead_of_differencing_it() -> None:
    objective = _Objective(_explosive_problem())
    # Locate the gain at which 150 steps overflow, then stand just below it.
    low, high = 0.0, 400.0
    for _ in range(80):
        middle = (low + high) / 2
        if objective.residuals(np.array([middle])) is None:
            high = middle
        else:
            low = middle
    edge = np.array([low])
    diagnosis, uncertainty = _identifiability(objective, edge)
    assert diagnosis == {"identifiable": False, "reason": "the model diverged near the optimum"}
    assert uncertainty["standard_errors"] is None
    beyond = _identifiability(objective, np.array([high + 1.0]))
    assert beyond[0]["reason"] == "the model diverged near the optimum"


def test_a_diverging_run_is_reported_as_none_and_held_out_as_diverged() -> None:
    assert simulate(_explosive(), "v", {"k": 300.0}, [0.0] * 150) is None
    problem = _explosive_problem()
    from sc_neurocore.fitting.fit import _rmse

    assert _rmse(problem, {"k": 300.0}, problem.holdout) == [
        {"recording": "flat again", "rmse": None, "diverged": True}
    ]
    # Finite but too large to square: no finite error either.
    assert _rmse(problem, {"k": 12.0}, problem.holdout) == [
        {"recording": "flat again", "rmse": None, "diverged": True}
    ]


def test_an_exported_fit_replays_to_the_same_digest(cohort: Any) -> None:
    result = fit_parameters(_problem(cohort), generations=3, population=5)
    replayed = replay_fit(result)
    assert replayed["reproduced"] is True
    assert replayed["replayed_sha256"] == replayed["exported_sha256"] == result["result_sha256"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"observable": "w"}, "'w' is not a state variable"),
        ({"schema": {"state": {"v": 0.0}}}, "declares no parameters"),
        ({"domains": ()}, "fit at least one parameter, each once"),
        (
            {"domains": (ParameterDomain("R", 0.1, 1.0), ParameterDomain("R", 0.1, 1.0))},
            "each once",
        ),
        ({"domains": (ParameterDomain("gain", 0.1, 1.0),)}, "gain is not a parameter"),
        ({"fixed": {"R": 1.0}}, "either fitted or fixed"),
        ({"holdout": ()}, "at least one training and one hold-out recording"),
    ],
)
def test_an_ill_formed_problem_is_refused(
    cohort: Any, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _problem(cohort, **change)


def test_the_same_name_or_data_in_both_splits_is_refused(cohort: Any) -> None:
    train, holdout = cohort
    renamed = Recording(holdout[0].name, train[0].current, train[0].observed)
    with pytest.raises(ValueError, match="recording names must be unique"):
        _problem(
            cohort, holdout=(Recording(train[0].name, holdout[0].current, holdout[0].observed),)
        )
    with pytest.raises(ValueError, match="appears in both the training and the hold-out set"):
        _problem(cohort, holdout=(renamed,))


@pytest.mark.parametrize(
    ("domain", "message"),
    [
        (("x", 1.0, 1.0, "linear"), "low < high"),
        (("x", 0.0, math.inf, "linear"), "low < high"),
        (("x", 0.0, 1.0, "log"), "must be positive"),
        (("x", 0.1, 1.0, "cubic"), "linear or log"),
    ],
)
def test_an_ill_formed_domain_is_refused(domain: tuple[Any, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ParameterDomain(*domain)


@pytest.mark.parametrize(
    ("current", "observed", "message"),
    [
        ((0.0,), (0.0,), ">= 2"),
        ((0.0, 1.0), (0.0,), "equal current and observed"),
        ((0.0, math.nan), (0.0, 1.0), "finite samples"),
    ],
)
def test_an_ill_formed_recording_is_refused(
    current: tuple[float, ...], observed: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        Recording("r", current, observed)


def test_another_fit_document_version_is_refused(cohort: Any) -> None:
    document = _problem(cohort).to_public_dict()
    document["schema_version"] = "sc-neurocore.fit.v0"
    with pytest.raises(ValueError, match="unsupported fit document"):
        problem_from_dict(document)


def test_the_optimiser_size_must_be_meaningful(cohort: Any) -> None:
    with pytest.raises(ValueError, match="generations must be at least 1"):
        fit_parameters(_problem(cohort), generations=0)
    with pytest.raises(ValueError, match="population at least 4"):
        fit_parameters(_problem(cohort), population=3)


def test_constrained_log_fit_is_feasible_replayable_and_has_no_unconstrained_error_bars(
    cohort: Any,
) -> None:
    """Value-space constraints survive logarithmic search and exact replay."""
    from sc_neurocore.fitting.constraints import ParameterConstraint

    problem = _problem(
        cohort,
        domains=(ParameterDomain("R", 0.1, 2.0, "log"),),
        fixed={"v_rest": -65.0, "tau_m": 10.0, "C": 1.0},
        constraints=(ParameterConstraint("R below C", {"R": 1.0, "C": -1.0}, -1.0, -0.3),),
    )
    events: list[dict[str, Any]] = []
    result = fit_parameters(problem, generations=12, population=8, progress=events.append)
    assert 0.65 < result["fitted"]["R"] <= 0.7
    assert result["optimiser"]["feasible"] is True
    assert result["optimiser"]["rejected_constraint_checks"] > 0
    assert result["uncertainty"]["standard_errors"] is None
    assert "constrained uncertainty" in result["uncertainty"]["reason"]
    assert result["problem"]["schema_version"] == "sc-neurocore.fit.v2"
    assert problem_from_dict(result["problem"]) == problem
    assert events == result["optimiser"]["history"]
    assert replay_fit(result)["reproduced"] is True


def test_infeasible_fit_never_claims_convergence_or_finite_training_loss(cohort: Any) -> None:
    """An empty feasible search remains an explicit failure with preserved history."""
    from sc_neurocore.fitting.constraints import ParameterConstraint

    problem = _problem(
        cohort,
        domains=(ParameterDomain("R", 0.1, 2.0),),
        fixed={"C": 1.0},
        constraints=(ParameterConstraint("impossible", {"R": 1.0}, 10.0, 20.0),),
    )
    result = fit_parameters(problem, generations=2, population=4)
    assert result["optimiser"]["feasible"] is False
    assert result["optimiser"]["converged"] is False
    assert result["training_loss"] is None
    assert all(row["loss"] is None for row in result["optimiser"]["history"])


def test_holdout_acquisition_groups_and_constraints_have_versioned_custody(cohort: Any) -> None:
    """Grouped recordings cannot cross the split even when their samples differ."""
    from dataclasses import replace
    from sc_neurocore.fitting.constraints import ParameterConstraint

    train, holdout = cohort
    grouped_train = tuple(replace(r, group="subject-a") for r in train)
    grouped_holdout = tuple(replace(r, group="subject-b") for r in holdout)
    problem = _problem((grouped_train, grouped_holdout))
    assert problem.to_public_dict()["schema_version"] == "sc-neurocore.fit.v2"
    assert problem_from_dict(problem.to_public_dict()) == problem
    with pytest.raises(ValueError, match="acquisition groups cannot cross"):
        _problem((grouped_train, tuple(replace(r, group="subject-a") for r in holdout)))
    with pytest.raises(ValueError, match="must be nonempty"):
        replace(train[0], group="")
    with pytest.raises(ValueError, match="fixed parameters must be finite"):
        _problem(cohort, fixed={"C": math.inf})
    constraint = ParameterConstraint("named", {"R": 1.0}, 0.0, 2.0)
    with pytest.raises(ValueError, match="constraint names must be unique"):
        _problem(cohort, constraints=(constraint, constraint))
    with pytest.raises(ValueError, match="unknown parameter"):
        _problem(cohort, constraints=(ParameterConstraint("bad", {"unknown": 1.0}, 0.0, 2.0),))
    document = _problem(cohort, constraints=(constraint,)).to_public_dict()
    document["schema_version"] = FIT_SCHEMA_VERSION
    with pytest.raises(ValueError, match="require a v2"):
        problem_from_dict(document)


@pytest.mark.parametrize(
    "arguments",
    [
        ("", {"R": 1.0}, 0.0, 2.0),
        ("no terms", {}, 0.0, 2.0),
        ("nonfinite", {"R": math.inf}, 0.0, 2.0),
        ("zero", {"R": 0.0}, 0.0, 2.0),
        ("equality", {"R": 1.0}, 1.0, 1.0),
        ("backwards", {"R": 1.0}, 2.0, 1.0),
    ],
)
def test_invalid_parameter_constraints_are_refused(arguments: tuple[Any, ...]) -> None:
    """Constraints refuse ambiguous or unsupported definitions before optimisation."""
    from sc_neurocore.fitting.constraints import ParameterConstraint

    with pytest.raises(ValueError):
        ParameterConstraint(*arguments)


def test_uncertainty_is_withheld_without_residual_degrees_of_freedom() -> None:
    """Two observations can identify two parameters without supporting a noise variance."""
    schema = _schema()
    training = simulate(schema, "v", TRUTH, [1.0, 2.0])
    held = simulate(schema, "v", TRUTH, [2.0, 3.0])
    assert training is not None and held is not None
    problem = FitProblem(
        schema,
        "v",
        (ParameterDomain("R", 0.1, 2.0), ParameterDomain("tau_m", 1.0, 30.0)),
        (Recording("train", (1.0, 2.0), tuple(training)),),
        (Recording("held", (2.0, 3.0), tuple(held)),),
        3,
        fixed={"v_rest": -65.0, "C": 1.0},
    )
    result = fit_parameters(problem, generations=5, population=4)
    assert result["identifiability"]["identifiable"] is True
    assert result["uncertainty"]["standard_errors"] is None
    assert "degrees of freedom" in result["uncertainty"]["reason"]


def test_domain_boundary_does_not_report_unconstrained_gaussian_error_bars(cohort: Any) -> None:
    """An optimum pressed against its domain has no ordinary Gaussian covariance claim."""
    result = fit_parameters(
        _problem(
            cohort,
            domains=(ParameterDomain("R", 0.1, 0.7),),
            fixed={"v_rest": -65.0, "tau_m": 10.0, "C": 1.0},
        ),
        generations=4,
        population=4,
    )
    assert result["fitted"]["R"] == pytest.approx(0.7)
    assert result["identifiability"]["identifiable"] is True
    assert result["uncertainty"]["standard_errors"] is None
    assert "boundary" in result["uncertainty"]["reason"]


def test_unrepresentable_local_sensitivity_is_reported_without_nan_covariance() -> None:
    """A finite low-loss model can have sensitivities too large for its Gram matrix."""
    schema = {
        "metadata": {"schema_version": 2, "name": "extreme units"},
        "state": {"v": 0.0},
        "parameters": {"k": 0.0, "gain": 1e200},
        "integration": {"dt": 1.0, "method": "euler"},
        "dynamics": {"v": "k * gain"},
    }
    problem = FitProblem(
        schema,
        "v",
        (ParameterDomain("k", -1e-125, 1e-125),),
        (Recording("train", (0.0,) * 4, (0.0,) * 4),),
        (Recording("held", (0.0,) * 4, (1.0,) * 4),),
        3,
    )
    result = fit_parameters(problem, generations=2, population=4)
    assert result["identifiability"]["identifiable"] is False
    assert result["identifiability"]["reason"] == "the residual Gram matrix is not finite"
    assert result["uncertainty"]["standard_errors"] is None


def test_fit_admits_schema_before_search_and_freezes_callback_mutations(cohort: Any) -> None:
    """Bad equations are refused before optimisation and caller-owned schemas stay independent."""
    schema = _schema()
    schema["dynamics"]["v"] = "__import__('os')"
    with pytest.raises(ValueError):
        _problem(cohort, schema=schema)
    problem = _problem(cohort)

    def edit_original(event: dict[str, Any]) -> None:
        problem.schema["parameters"]["C"] = 2.0

    result = fit_parameters(problem, generations=2, population=4, progress=edit_original)
    assert result["problem"]["schema"]["parameters"]["C"] == 1.0
    assert replay_fit(result)["reproduced"] is True


@pytest.mark.parametrize("seed", [-1, 2**32, True, 0.5])
def test_fit_seeds_are_admitted_without_rounding_or_late_worker_failure(
    cohort: Any, seed: Any
) -> None:
    """The legacy SciPy seed space is checked before optimisation or replay."""
    with pytest.raises(ValueError, match="fit seed must be an integer"):
        _problem(cohort, seed=seed)
    document = _problem(cohort).to_public_dict()
    document["seed"] = seed
    with pytest.raises(ValueError, match="fit seed must be an integer"):
        problem_from_dict(document)
