# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_alternative_path.py

from __future__ import annotations


import numpy as np


import pytest


from sc_neurocore.experimental import solver_routes


from sc_neurocore.experimental import (
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRegistry,
    AlternativePathRoute,
    build_builtin_registry,
    build_demo_registry,
    default_report_path,
    make_delayed_recall_shared_state_route,
    make_harmonic_symplectic_route,
    make_heat_cosine_mode_route,
    make_kuramoto_noiseless_symplectic_lift_route,
    make_lif_subthreshold_exact_route,
    write_batch_report,
)


from sc_neurocore.experimental.alternative_path import compare_outputs


from sc_neurocore.experimental.builtins import builtin_cases_for_route


def _broken_shadow_route():
    return AlternativePathRoute(
        name="safe.shadow-broken",
        baseline=lambda: 1.0,
        candidate=lambda: (_ for _ in ()).throw(RuntimeError("shadow boom")),
        summary="Shadow route whose candidate raises",
        expected_behavior="Returns the baseline and records the candidate error",
    )


__all__ = [
    "np",
    "pytest",
    "solver_routes",
    "AlternativePathCase",
    "AlternativePathConfig",
    "AlternativePathMode",
    "AlternativePathRegistry",
    "AlternativePathRoute",
    "build_builtin_registry",
    "build_demo_registry",
    "default_report_path",
    "make_delayed_recall_shared_state_route",
    "make_harmonic_symplectic_route",
    "make_heat_cosine_mode_route",
    "make_kuramoto_noiseless_symplectic_lift_route",
    "make_lif_subthreshold_exact_route",
    "write_batch_report",
    "compare_outputs",
    "builtin_cases_for_route",
    "_broken_shadow_route",
]
