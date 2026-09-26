# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Generate an explicit synthetic Studio experiment cohort

"""Export a synthetic shared-noise LIF cohort for Studio's experiment laboratory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np

from sc_neurocore.fitting import (
    CohortMetric,
    CohortModel,
    CohortSample,
    ExperimentCohort,
    SweepDomain,
    run_cohort,
)
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema


def synthetic_cohort(seed: int = 7) -> ExperimentCohort:
    """Record synthetic truth once and share its explicit stimuli across all models.

    Parameters
    ----------
    seed:
        Generator seed used only when creating the exported input-noise samples.

    Returns
    -------
    ExperimentCohort
        Two independent acquisitions, two model-specific metric contracts and
        every parameter value. These are synthetic data, not hardware evidence.
    """
    rng = np.random.default_rng(seed)
    schema = load_schema("lif")
    samples: list[CohortSample] = []
    entries: tuple[tuple[str, float, Literal["train", "holdout"]], ...] = (
        ("step-8", 8.0, "train"),
        ("step-20", 20.0, "holdout"),
    )
    for name, level, split in entries:
        current = (0.0,) * 20 + (level,) * 80
        noise = tuple(float(v) for v in rng.normal(0.0, 0.2, len(current)))
        neuron = UniversalNeuron.from_dict(schema)
        values: list[float] = []
        spikes: list[int] = []
        for drive, jitter in zip(current, noise, strict=True):
            spikes.append(neuron.step(I=drive + jitter))
            values.append(float(neuron.state["v"]))
        samples.append(
            CohortSample(name, name, split, current, noise, {"v": tuple(values)}, tuple(spikes))
        )
    domains = (SweepDomain("R", (0.5, 1.0, 1.5)),)
    models = (
        CohortModel("voltage-error", schema, domains, CohortMetric("trace_rmse", "v", "mV")),
        CohortModel(
            "event-error", schema, domains, CohortMetric("event_disagreement", "", "fraction")
        ),
    )
    return ExperimentCohort(
        "synthetic LIF shared noise",
        tuple(samples),
        models,
        1.0,
        "ms",
        "nA",
        seed,
        f"NumPy {np.__version__} default_rng({seed}), normal(0,0.2 nA); samples exported exactly",
    )


def main() -> None:
    """Write a new cohort or its complete result without overwriting an experiment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--run",
        action="store_true",
        help="export the complete sweep result instead of only the protocol",
    )
    args = parser.parse_args()
    cohort = synthetic_cohort(args.seed)
    document = run_cohort(cohort) if args.run else cohort.to_public_dict()
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")


if __name__ == "__main__":
    main()
