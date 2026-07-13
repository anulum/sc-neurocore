# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio policy architecture tests

"""Compatibility and responsibility-boundary tests for Studio policy."""

from __future__ import annotations

import ast
import pickle
from pathlib import Path
from types import ModuleType

from sc_neurocore.studio.platform import policy as facade
from sc_neurocore.studio.platform import policy_audit
from sc_neurocore.studio.platform import policy_gateway
from sc_neurocore.studio.platform import policy_models
from sc_neurocore.studio.platform import policy_routes
from sc_neurocore.studio.platform.policy_routes_compute import COMPUTE_ROUTES
from sc_neurocore.studio.platform.policy_routes_discovery import DISCOVERY_ROUTES
from sc_neurocore.studio.platform.policy_routes_platform import PLATFORM_ROUTES
from sc_neurocore.studio.platform.policy_routes_workspace import WORKSPACE_ROUTES


def _module_path(module: ModuleType) -> Path:
    """Return the concrete source path for a loaded policy module."""

    assert module.__file__ is not None
    return Path(module.__file__)


def test_historical_facade_reexports_identical_public_objects() -> None:
    """Every historical public import resolves to its single owning object."""

    expected = {
        "AuditEvent": policy_models.AuditEvent,
        "AuditExport": policy_models.AuditExport,
        "AuditQuarantineExport": policy_models.AuditQuarantineExport,
        "AuditSink": policy_models.AuditSink,
        "AuditSinkError": policy_models.AuditSinkError,
        "AuditSinkStatus": policy_models.AuditSinkStatus,
        "InMemoryAuditSink": policy_models.InMemoryAuditSink,
        "JsonlAuditSink": policy_audit.JsonlAuditSink,
        "PolicyDecision": policy_models.PolicyDecision,
        "PolicyGateway": policy_gateway.PolicyGateway,
        "Principal": policy_models.Principal,
        "RoutePolicy": policy_models.RoutePolicy,
        "RoutePolicyRegistry": policy_gateway.RoutePolicyRegistry,
        "RouteVisibility": policy_models.RouteVisibility,
        "build_default_studio_route_policy_registry": (
            policy_routes.build_default_studio_route_policy_registry
        ),
    }
    for name, implementation in expected.items():
        exported = getattr(facade, name)
        assert exported is implementation
        assert exported.__module__ == facade.__name__


def test_historical_pickle_paths_round_trip() -> None:
    """Representative public value objects retain the historical pickle path."""

    principal = facade.Principal(
        principal_id="architecture-test",
        roles=frozenset({"studio.viewer"}),
    )
    restored = pickle.loads(pickle.dumps(principal))
    assert restored == principal
    assert type(restored) is facade.Principal


def test_route_catalogues_form_one_duplicate_free_registry() -> None:
    """Cohesive catalogues preserve the complete route set and stable order."""

    routes = (
        *PLATFORM_ROUTES,
        *DISCOVERY_ROUTES,
        *COMPUTE_ROUTES,
        *WORKSPACE_ROUTES,
    )
    signatures = [(method.upper(), path) for method, path, _visibility, _action in routes]
    registry = facade.build_default_studio_route_policy_registry()
    registered = [(method, path) for method, path, _policy in registry.policies()]
    assert len(signatures) == len(set(signatures))
    assert set(registered) == set(signatures)


def test_responsibility_modules_remain_bounded() -> None:
    """The split must not collapse back into a monolithic implementation."""

    modules = (
        facade,
        policy_audit,
        policy_gateway,
        policy_models,
        policy_routes,
    )
    route_modules = (
        "policy_routes_compute.py",
        "policy_routes_discovery.py",
        "policy_routes_platform.py",
        "policy_routes_workspace.py",
    )
    paths = [_module_path(module) for module in modules]
    paths.extend(Path(facade.__file__).with_name(name) for name in route_modules)
    assert max(len(path.read_text(encoding="utf-8").splitlines()) for path in paths) <= 300


def test_focused_policy_test_modules_remain_bounded() -> None:
    """Focused tests must not be recombined into another test GodFile."""

    test_root = Path(__file__).parent
    paths = sorted(test_root.glob("test_studio_policy_*.py"))
    paths.append(test_root / "studio_policy_support.py")
    assert paths
    assert max(len(path.read_text(encoding="utf-8").splitlines()) for path in paths) <= 300


def test_policy_responsibility_graph_is_acyclic() -> None:
    """Implementation modules must not depend back on the historical facade."""

    package_root = Path(facade.__file__).parent
    paths = sorted(package_root.glob("policy*.py"))
    module_names = {path.stem for path in paths}
    dependencies: dict[str, set[str]] = {}
    prefix = "sc_neurocore.studio.platform."
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        dependencies[path.stem] = {
            node.module.removeprefix(prefix)
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith(prefix)
            and node.module.removeprefix(prefix) in module_names
        }

    assert all("policy" not in deps for name, deps in dependencies.items() if name != "policy")
    pending = {name: set(deps) for name, deps in dependencies.items()}
    while pending:
        ready = {name for name, deps in pending.items() if not deps.intersection(pending)}
        assert ready, f"cyclic Studio policy imports: {pending}"
        for name in ready:
            pending.pop(name)
