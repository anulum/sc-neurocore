# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio responsibility-router contract tests

"""Verify Studio route ownership and the committed OpenAPI contract."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.routing import APIRoute, APIWebSocketRoute
from starlette.testclient import TestClient

from sc_neurocore.studio.api.frontend import mount_studio_frontend
from sc_neurocore.studio.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_REFERENCE = REPO_ROOT / "docs" / "_generated" / "studio_openapi.json"
EXPECTED_HTTP_ROUTE_MODULES = frozenset(
    {
        "sc_neurocore.studio.api.adaptive_precision",
        "sc_neurocore.studio.api.audit",
        "sc_neurocore.studio.api.catalogue",
        "sc_neurocore.studio.api.compiler",
        "sc_neurocore.studio.api.cosim",
        "sc_neurocore.studio.api.deploy",
        "sc_neurocore.studio.api.design",
        "sc_neurocore.studio.api.export",
        "sc_neurocore.studio.api.identity",
        "sc_neurocore.studio.api.jobs",
        "sc_neurocore.studio.api.presets",
        "sc_neurocore.studio.api.simulation",
        "sc_neurocore.studio.api.synthesis",
        "sc_neurocore.studio.api.system",
        "sc_neurocore.studio.api.training",
        "sc_neurocore.studio.api.training_weights",
    }
)


def _leaf_routes(routes: Iterable[Any]) -> Iterator[Any]:
    """Yield leaf routes, descending into Starlette 1.3 included sub-routers.

    Starlette 1.3 wraps ``include_router`` routes in an ``_IncludedRouter`` whose
    leaf routes live under ``original_router.routes`` instead of being flattened
    onto ``app.routes``; older Starlette flattened them. Walk both shapes.
    """
    for route in routes:
        original_router = getattr(route, "original_router", None)
        if original_router is not None and hasattr(original_router, "routes"):
            yield from _leaf_routes(original_router.routes)
        elif getattr(route, "routes", None):
            yield from _leaf_routes(route.routes)
        else:
            yield route


def test_application_routes_are_owned_by_responsibility_modules() -> None:
    application = create_app()
    routes = [route for route in _leaf_routes(application.routes) if isinstance(route, APIRoute)]
    backend_routes = [route for route in routes if route.path != "/"]
    root_routes = [route for route in routes if route.path == "/"]
    signatures = [(route.path, tuple(sorted(route.methods or ()))) for route in routes]

    assert len(backend_routes) == 117
    assert {route.endpoint.__module__ for route in backend_routes} == EXPECTED_HTTP_ROUTE_MODULES
    assert len(root_routes) <= 1
    assert all(
        route.endpoint.__module__ == "sc_neurocore.studio.api.frontend" for route in root_routes
    )
    assert len(signatures) == len(set(signatures))
    assert all(route.endpoint.__module__ != "sc_neurocore.studio.app" for route in routes)

    websocket_routes = [
        route for route in _leaf_routes(application.routes) if isinstance(route, APIWebSocketRoute)
    ]
    assert [(route.path, route.endpoint.__module__) for route in websocket_routes] == [
        ("/ws/progress", "sc_neurocore.studio.api.export")
    ]


def test_runtime_openapi_matches_committed_reference() -> None:
    committed = json.loads(OPENAPI_REFERENCE.read_text(encoding="utf-8"))

    assert create_app().openapi() == committed


def test_frontend_mount_supports_source_tree_fallback(tmp_path: Path) -> None:
    app_module_file = tmp_path / "one" / "two" / "three" / "four" / "app.py"
    app_module_file.parent.mkdir(parents=True)
    fallback_dist = tmp_path / "studio" / "frontend" / "dist"
    fallback_dist.mkdir(parents=True)
    (fallback_dist / "index.html").write_text("<html>fallback</html>", encoding="utf-8")
    application = FastAPI()

    mount_studio_frontend(application, app_module_file=str(app_module_file))
    response = TestClient(application, base_url="http://127.0.0.1").get("/")

    assert response.status_code == 200
    assert response.text == "<html>fallback</html>"


def test_frontend_mount_leaves_root_unclaimed_without_distribution(tmp_path: Path) -> None:
    app_module_file = tmp_path / "one" / "two" / "three" / "four" / "app.py"
    app_module_file.parent.mkdir(parents=True)
    application = FastAPI()

    mount_studio_frontend(application, app_module_file=str(app_module_file))
    response = TestClient(application, base_url="http://127.0.0.1").get("/")

    assert response.status_code == 404
