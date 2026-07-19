# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio public discovery route policies

"""Public discovery route-policy catalogue."""

from __future__ import annotations

from sc_neurocore.studio.platform.policy_models import RouteVisibility

DISCOVERY_ROUTES: tuple[tuple[str, str, RouteVisibility, str], ...] = (
    ("GET", "/api/templates", RouteVisibility.PUBLIC, "studio.templates.read"),
    ("GET", "/api/templates/{name}", RouteVisibility.PUBLIC, "studio.templates.read"),
    ("GET", "/api/models", RouteVisibility.PUBLIC, "studio.models.read"),
    ("GET", "/api/models/{name}", RouteVisibility.PUBLIC, "studio.models.read"),
    ("GET", "/api/models/scan", RouteVisibility.PUBLIC, "studio.models.scan"),
    (
        "POST",
        "/api/models/scan/jobs",
        RouteVisibility.AUTHENTICATED,
        "studio.models.scan.job",
    ),
    ("GET", "/api/models/facets", RouteVisibility.PUBLIC, "studio.models.read"),
    ("GET", "/api/models/{name}/doc", RouteVisibility.PUBLIC, "studio.models.read"),
    ("GET", "/api/dcls/info", RouteVisibility.PUBLIC, "studio.dcls.read"),
    ("GET", "/api/dcls/benchmark", RouteVisibility.PUBLIC, "studio.dcls.benchmark"),
    ("POST", "/api/dcls/evaluate", RouteVisibility.PUBLIC, "studio.dcls.evaluate"),
    ("GET", "/api/benchmarks/schema", RouteVisibility.PUBLIC, "studio.benchmarks.read"),
    ("POST", "/api/benchmarks/run", RouteVisibility.PUBLIC, "studio.benchmarks.run"),
    (
        "POST",
        "/api/benchmarks/contribute",
        RouteVisibility.PUBLIC,
        "studio.benchmarks.contribute",
    ),
    ("GET", "/api/benchmarks/databank", RouteVisibility.PUBLIC, "studio.benchmarks.read"),
    ("GET", "/api/presets", RouteVisibility.PUBLIC, "studio.presets.read"),
    (
        "GET",
        "/api/presets/{preset_id}",
        RouteVisibility.PUBLIC,
        "studio.presets.read",
    ),
    (
        "GET",
        "/api/presets/actions/catalog",
        RouteVisibility.PUBLIC,
        "studio.presets.actions.catalog",
    ),
    (
        "GET",
        "/api/presets/{preset_id}/actions",
        RouteVisibility.PUBLIC,
        "studio.presets.actions.read",
    ),
    (
        "GET",
        "/api/presets/{preset_id}/default-flow/plan",
        RouteVisibility.PUBLIC,
        "studio.presets.default_flow.plan.read",
    ),
    (
        "GET",
        "/api/presets/{preset_id}/default-flow/contract",
        RouteVisibility.PUBLIC,
        "studio.presets.default_flow.contract.read",
    ),
    ("GET", "/api/cache/stats", RouteVisibility.PUBLIC, "studio.cache.stats.read"),
    (
        "GET",
        "/api/synth/tools-status",
        RouteVisibility.PUBLIC,
        "studio.synth.tools.read",
    ),
    (
        "GET",
        "/api/training/surrogates",
        RouteVisibility.PUBLIC,
        "studio.training.surrogates.read",
    ),
    (
        "GET",
        "/api/training/cell-types",
        RouteVisibility.PUBLIC,
        "studio.training.cell_types.read",
    ),
    ("GET", "/api/graph/models", RouteVisibility.PUBLIC, "studio.graph.models.read"),
)
