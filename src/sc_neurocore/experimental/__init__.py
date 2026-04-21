# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.experimental -- Safe opt-in harness for alternative paths

"""sc_neurocore.experimental -- Tier: experimental.

This package provides an explicit, non-destructive harness for evaluating
alternative implementations against a stable baseline. Nothing here is active
unless imported and enabled via ``AlternativePathConfig``.
"""

__tier__ = "experimental"

from .alternative_path import (
    AlternativePathBatchSummary,
    AlternativePathCase,
    AlternativePathConfig,
    AlternativePathMode,
    AlternativePathRegistry,
    AlternativePathResult,
    AlternativePathRoute,
    ComparisonStats,
)
from .builtins import build_builtin_registry
from .examples import build_demo_registry, make_demo_sigmoid_route
from .physics_routes import (
    make_harmonic_symplectic_route,
    make_heat_cosine_mode_route,
    make_kuramoto_noiseless_symplectic_lift_route,
)
from .reporting import default_report_path, write_batch_report
from .solver_routes import make_lif_subthreshold_exact_route

__all__ = [
    "AlternativePathBatchSummary",
    "AlternativePathCase",
    "AlternativePathConfig",
    "AlternativePathMode",
    "AlternativePathRegistry",
    "AlternativePathResult",
    "AlternativePathRoute",
    "ComparisonStats",
    "build_builtin_registry",
    "build_demo_registry",
    "default_report_path",
    "make_harmonic_symplectic_route",
    "make_heat_cosine_mode_route",
    "make_kuramoto_noiseless_symplectic_lift_route",
    "make_lif_subthreshold_exact_route",
    "make_demo_sigmoid_route",
    "write_batch_report",
]
