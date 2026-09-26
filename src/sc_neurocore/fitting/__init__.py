# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Parameter fitting with held-out validation and identifiability

"""Fit Universal DSL model parameters to recordings, and state what the data support."""

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
from sc_neurocore.fitting.pareto import measured_pareto
from sc_neurocore.fitting.fit import (
    IDENTIFIABILITY_RATIO,
    UNCERTAINTY_METHOD,
    fit_parameters,
    replay_fit,
)
from sc_neurocore.fitting.problem import (
    FIT_SCHEMA_VERSION,
    FitProblem,
    ParameterDomain,
    Recording,
    problem_from_dict,
    simulate,
)

__all__ = [
    "CohortMetric",
    "CohortModel",
    "CohortSample",
    "ExperimentCohort",
    "SweepDomain",
    "ParameterConstraint",
    "cohort_from_dict",
    "cohort_sha256",
    "run_cohort",
    "replay_cohort",
    "measured_pareto",
    "FIT_SCHEMA_VERSION",
    "FitProblem",
    "IDENTIFIABILITY_RATIO",
    "ParameterDomain",
    "Recording",
    "UNCERTAINTY_METHOD",
    "fit_parameters",
    "problem_from_dict",
    "replay_fit",
    "simulate",
]
