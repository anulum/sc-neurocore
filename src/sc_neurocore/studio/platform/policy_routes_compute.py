# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compute and graph route policies

"""Compute and graph route-policy catalogue."""

from __future__ import annotations

from sc_neurocore.studio.platform.policy_models import RouteVisibility

COMPUTE_ROUTES: tuple[tuple[str, str, RouteVisibility, str], ...] = (
    (
        "POST",
        "/api/simulate",
        RouteVisibility.AUTHENTICATED,
        "studio.simulation.run",
    ),
    (
        "POST",
        "/api/models/simulate",
        RouteVisibility.AUTHENTICATED,
        "studio.models.simulate",
    ),
    (
        "POST",
        "/api/multi-simulate",
        RouteVisibility.AUTHENTICATED,
        "studio.models.multi_simulate",
    ),
    (
        "POST",
        "/api/characterize",
        RouteVisibility.AUTHENTICATED,
        "studio.models.characterize",
    ),
    (
        "POST",
        "/api/classify",
        RouteVisibility.AUTHENTICATED,
        "studio.models.classify",
    ),
    ("POST", "/api/fi-curve", RouteVisibility.AUTHENTICATED, "studio.analysis.fi_curve"),
    (
        "POST",
        "/api/bifurcation",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.bifurcation",
    ),
    (
        "POST",
        "/api/sensitivity",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.sensitivity",
    ),
    (
        "POST",
        "/api/nullclines",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.nullclines",
    ),
    ("POST", "/api/precision", RouteVisibility.AUTHENTICATED, "studio.analysis.precision"),
    (
        "POST",
        "/api/freq-response",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.freq_response",
    ),
    ("POST", "/api/heatmap", RouteVisibility.AUTHENTICATED, "studio.analysis.heatmap"),
    (
        "POST",
        "/api/analysis/jobs",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.job",
    ),
    (
        "POST",
        "/api/compare",
        RouteVisibility.AUTHENTICATED,
        "studio.analysis.compare",
    ),
    (
        "POST",
        "/api/import-trace",
        RouteVisibility.AUTHENTICATED,
        "studio.trace.import",
    ),
    ("POST", "/api/codegen", RouteVisibility.AUTHENTICATED, "studio.codegen.generate"),
    ("POST", "/api/compile", RouteVisibility.AUTHENTICATED, "studio.compiler.compile"),
    (
        "POST",
        "/api/models/compile",
        RouteVisibility.AUTHENTICATED,
        "studio.compiler.model_compile",
    ),
    (
        "POST",
        "/api/models/cosim",
        RouteVisibility.AUTHENTICATED,
        "studio.compiler.model_cosim",
    ),
    (
        "POST",
        "/api/nir/compile",
        RouteVisibility.AUTHENTICATED,
        "studio.compiler.nir_compile",
    ),
    (
        "POST",
        "/api/adaptive-precision/auto-tune",
        RouteVisibility.AUTHENTICATED,
        "studio.compiler.adaptive_precision.auto_tune",
    ),
    (
        "POST",
        "/api/adaptive-precision/formal-bundle",
        RouteVisibility.AUTHENTICATED,
        "studio.compiler.adaptive_precision.formal_bundle",
    ),
    ("POST", "/api/ir/build", RouteVisibility.AUTHENTICATED, "studio.ir.build"),
    ("POST", "/api/ir/verify", RouteVisibility.AUTHENTICATED, "studio.ir.verify"),
    ("POST", "/api/ir/emit-sv", RouteVisibility.AUTHENTICATED, "studio.ir.emit_sv"),
    (
        "POST",
        "/api/ir/emit-sv-direct",
        RouteVisibility.AUTHENTICATED,
        "studio.ir.emit_sv_direct",
    ),
    ("POST", "/api/ir/cosim", RouteVisibility.AUTHENTICATED, "studio.ir.cosim"),
    (
        "POST",
        "/api/synth/estimate",
        RouteVisibility.AUTHENTICATED,
        "studio.synth.estimate",
    ),
    ("POST", "/api/synth/run", RouteVisibility.ADMIN, "studio.synth.run"),
    (
        "POST",
        "/api/synth/terminal",
        RouteVisibility.ADMIN,
        "studio.synth.terminal",
    ),
    (
        "POST",
        "/api/synth/multi-target",
        RouteVisibility.ADMIN,
        "studio.synth.multi_target",
    ),
    ("POST", "/api/synth/pnr", RouteVisibility.ADMIN, "studio.synth.pnr"),
    ("POST", "/api/export/svg", RouteVisibility.AUTHENTICATED, "studio.export.svg"),
    (
        "POST",
        "/api/network/ei",
        RouteVisibility.AUTHENTICATED,
        "studio.network.ei.simulate",
    ),
    (
        "POST",
        "/api/graph/population",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.population.create",
    ),
    (
        "POST",
        "/api/graph/projection",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.projection.create",
    ),
    (
        "POST",
        "/api/graph/validate",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.validate",
    ),
    (
        "POST",
        "/api/graph/simulate",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.simulate",
    ),
    (
        "POST",
        "/api/graph/export-nir",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.export_nir",
    ),
    (
        "POST",
        "/api/graph/import-nir",
        RouteVisibility.AUTHENTICATED,
        "studio.graph.import_nir",
    ),
)
