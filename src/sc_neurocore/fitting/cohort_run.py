# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Replayable cohort execution and comparison

"""Run full declared sweeps with exact common samples and visible failures."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from typing import Any

from sc_neurocore.fitting.cohort import (
    CohortModel,
    CohortSample,
    ExperimentCohort,
    cohort_from_dict,
)
from sc_neurocore.fitting.cohort import cohort_sha256


def _measure(
    model: CohortModel, sample: CohortSample, parameters: dict[str, float]
) -> float | None:
    """Score one complete trial using the existing public DSL step function."""
    from sc_neurocore.neurons.universal_dsl import UniversalNeuron

    neuron = UniversalNeuron.from_dict(dict(model.schema), parameter_overrides=parameters)
    squared = 0.0
    count = 0
    disagreements = 0
    for index, current in enumerate(sample.effective_current):
        try:
            spike = neuron.step(I=current)
        except (FloatingPointError, OverflowError, ZeroDivisionError):
            return None
        count += spike
        if model.metric.kind != "trace_rmse":
            disagreements += int(spike != sample.spikes[index])
        if model.metric.kind == "trace_rmse":
            residual = float(neuron.state[model.metric.observable]) - float(
                sample.observations[model.metric.observable][index]
            )
            squared += residual * residual
    if model.metric.kind == "trace_rmse":
        value = math.sqrt(squared / len(sample.current))
    elif model.metric.kind == "event_disagreement":
        value = disagreements / len(sample.current)
    else:
        value = float(abs(count - sum(sample.spikes)))
    return value if math.isfinite(value) else None


def run_cohort(
    cohort: ExperimentCohort, *, progress: Callable[[dict[str, Any]], None] | None = None
) -> dict[str, Any]:
    """Run every declared trial, retaining rejected and divergent members.

    Parameters
    ----------
    cohort:
        Admitted complete models, samples, sweep domains and split custody.
    progress:
        Optional trial-event sink; raising cancels execution.

    Returns
    -------
    dict
        Full cohort, all trial outcomes, training-only selection, provenance
        and deterministic result digest. Hardware measurements are not inferred.
    """
    cohort = cohort_from_dict(json.loads(json.dumps(cohort.to_public_dict(), allow_nan=False)))
    trials: list[dict[str, Any]] = []
    for model in cohort.models:
        for coordinates in model.parameter_sets():
            parameters = {
                **dict(model.schema["parameters"]),
                **dict(model.fixed),
                **dict(zip((d.name for d in model.domains), coordinates, strict=True)),
            }
            refused = [c.name for c in model.constraints if not c.accepts(parameters)]
            rows: list[dict[str, Any]] = []
            if not refused:
                for sample in cohort.samples:
                    value = _measure(model, sample, parameters)
                    rows.append(
                        {
                            "sample": sample.name,
                            "split": sample.split,
                            "value": value,
                            "failed": value is None,
                        }
                    )
            status = (
                "constraint_rejected"
                if refused
                else "failed"
                if any(row["failed"] for row in rows)
                else "completed"
            )
            metric = model.metric.to_public_dict()
            body = {
                "model": model.name,
                "parameters": parameters,
                "metric": metric,
                "status": status,
                "rejected_constraints": refused,
                "samples": rows,
                "schema_sha256": cohort_sha256(dict(model.schema)),
            }
            body["trial_sha256"] = cohort_sha256(body)
            trials.append(body)
            if progress is not None:
                progress({"trial": len(trials), "total": cohort.trial_count, "status": status})
    selection: list[dict[str, Any]] = []
    for model in cohort.models:
        candidates = [
            trial
            for trial in trials
            if trial["model"] == model.name
            and not trial["rejected_constraints"]
            and all(not row["failed"] for row in trial["samples"] if row["split"] == "train")
        ]
        if candidates:

            def training_error(trial: dict[str, Any]) -> float:
                values = [row["value"] for row in trial["samples"] if row["split"] == "train"]
                return float(sum(values) / len(values))

            best = min(candidates, key=training_error)
            selection.append(
                {
                    "model": model.name,
                    "trial_sha256": best["trial_sha256"],
                    "training_metric": training_error(best),
                    "selection_split": "train",
                }
            )
        else:
            selection.append(
                {"model": model.name, "trial_sha256": None, "reason": "no complete feasible trial"}
            )
    from sc_neurocore import __version__
    import numpy as np

    document = cohort.to_public_dict()
    result: dict[str, Any] = {
        "schema_version": "sc-neurocore.cohort-result.v1",
        "cohort": document,
        "trials": trials,
        "selection": selection,
        "provenance": {
            "cohort_sha256": cohort_sha256(document),
            "sample_sha256": {s.name: cohort_sha256(s.to_public_dict()) for s in cohort.samples},
            "seed": cohort.seed,
            "noise_provenance": cohort.noise_provenance,
            "sc_neurocore": __version__,
            "numpy": np.__version__,
            "selection_weighting": "arithmetic mean of per-recording training metrics",
        },
        "measurement_status": "not measured; no latency/resource/energy Pareto claim",
    }
    result["result_sha256"] = cohort_sha256(result)
    return result


def replay_cohort(result: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the whole sweep and compare complete result digests."""
    again = run_cohort(cohort_from_dict(result["cohort"]))
    return {
        "reproduced": again["result_sha256"] == result["result_sha256"],
        "exported_sha256": result["result_sha256"],
        "replayed_sha256": again["result_sha256"],
        "result": again,
    }
