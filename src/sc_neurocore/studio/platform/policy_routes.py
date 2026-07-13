# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio route registry assembly

"""Assembly of the complete default Studio route-policy registry."""

from __future__ import annotations

from sc_neurocore.studio.platform.policy_gateway import RoutePolicyRegistry
from sc_neurocore.studio.platform.policy_models import RoutePolicy, RouteVisibility
from sc_neurocore.studio.platform.policy_routes_compute import COMPUTE_ROUTES
from sc_neurocore.studio.platform.policy_routes_discovery import DISCOVERY_ROUTES
from sc_neurocore.studio.platform.policy_routes_platform import PLATFORM_ROUTES
from sc_neurocore.studio.platform.policy_routes_workspace import WORKSPACE_ROUTES

RouteDefinition = tuple[str, str, RouteVisibility, str]


def _register_routes(
    registry: RoutePolicyRegistry,
    routes: tuple[RouteDefinition, ...],
) -> None:
    """Register a cohesive route-policy catalogue."""

    for method, path_template, visibility, audit_action in routes:
        registry.register(
            method,
            path_template,
            RoutePolicy(visibility=visibility, audit_action=audit_action),
        )


def build_default_studio_route_policy_registry() -> RoutePolicyRegistry:
    """Build route policies for the current Studio platform API surface."""

    registry = RoutePolicyRegistry()
    for routes in (
        PLATFORM_ROUTES,
        DISCOVERY_ROUTES,
        COMPUTE_ROUTES,
        WORKSPACE_ROUTES,
    ):
        _register_routes(registry, routes)
    return registry
