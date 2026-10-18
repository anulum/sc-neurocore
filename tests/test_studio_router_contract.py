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

from sc_neurocore.studio.api.frontend import mount_studio_frontend, studio_entry
from sc_neurocore.studio.app import create_app

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENAPI_REFERENCE = REPO_ROOT / "docs" / "_generated" / "studio_openapi.json"
EXPECTED_HTTP_ROUTE_MODULES = frozenset(
    {
        "sc_neurocore.studio.api.adaptive_precision",
        "sc_neurocore.studio.api.audit",
        "sc_neurocore.studio.api.candidates",
        "sc_neurocore.studio.api.catalogue",
        "sc_neurocore.studio.api.compiler",
        "sc_neurocore.studio.api.cosim",
        "sc_neurocore.studio.api.deploy",
        "sc_neurocore.studio.api.design",
        "sc_neurocore.studio.api.export",
        "sc_neurocore.studio.api.fit_jobs",
        "sc_neurocore.studio.api.fits",
        "sc_neurocore.studio.api.frontend",
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
    """Bind every declared route to its expected responsibility module."""
    application = create_app()
    routes = [route for route in _leaf_routes(application.routes) if isinstance(route, APIRoute)]
    backend_routes = [route for route in routes if route.path != "/"]
    root_routes = [route for route in routes if route.path == "/"]
    signatures = [(route.path, tuple(sorted(route.methods or ()))) for route in routes]

    # 140 since `POST /api/graph/notebook`, a tutorial notebook that rebuilds
    # a drawn network with the public API and checks it against the Studio run.
    # 148 since the fit and cohort job routes (`/api/fits/jobs`, its status and
    # cancel, `/api/fits/replay/jobs`, `/api/cohorts/jobs`, `/measurements`,
    # `/replay`) and `GET /api/training/target-profiles`, which landed without
    # this count being raised, so the assertion failed on main until then.
    # The count is pinned so a route cannot
    # appear without a deliberate change here; raising it is how a new route is
    # admitted, never by relaxing the assertion.
    # Swagger and ReDoc now have explicit frontend-owned handlers so their
    # generated scripts can be bound to the response CSP by content hash.
    assert len(backend_routes) == 150
    assert {
        route.path
        for route in backend_routes
        if route.endpoint.__module__ == "sc_neurocore.studio.api.frontend"
    } == {"/docs", "/redoc"}
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
    """Match the runtime API schema to its maintained generated reference."""
    committed = json.loads(OPENAPI_REFERENCE.read_text(encoding="utf-8"))

    assert create_app().openapi() == committed


def test_frontend_mount_supports_source_tree_fallback(tmp_path: Path) -> None:
    """Serve the source-tree frontend when no packaged frontend is available."""
    app_module_file = tmp_path / "one" / "two" / "three" / "four" / "app.py"
    app_module_file.parent.mkdir(parents=True)
    fallback_dist = tmp_path / "studio" / "frontend" / "dist"
    fallback_dist.mkdir(parents=True)
    (fallback_dist / "index.html").write_text("<html>fallback</html>", encoding="utf-8")
    (fallback_dist / "assets").mkdir()
    (fallback_dist / "assets" / "app.js").write_text("console.log(1)", encoding="utf-8")
    application = FastAPI()

    mount_studio_frontend(application, app_module_file=str(app_module_file))
    client = TestClient(
        application, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )
    root = client.get("/", follow_redirects=False)
    page = client.get("/studios/sc-neurocore/")
    asset = client.get("/studios/sc-neurocore/assets/app.js")

    # The build names its published path in every asset URL, so it is served there.
    assert root.status_code == 307
    assert root.headers["location"] == "/studios/sc-neurocore/"
    assert page.status_code == 200
    assert page.text == "<html>fallback</html>"
    assert asset.status_code == 200


def test_frontend_mount_leaves_root_unclaimed_without_distribution(tmp_path: Path) -> None:
    """Leave the root unclaimed when no built frontend exists."""
    app_module_file = tmp_path / "one" / "two" / "three" / "four" / "app.py"
    app_module_file.parent.mkdir(parents=True)
    application = FastAPI()

    mount_studio_frontend(application, app_module_file=str(app_module_file))
    response = TestClient(
        application, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ).get("/")

    assert response.status_code == 404


def test_a_launched_studio_opens_its_ui_or_says_there_is_none(tmp_path: Path) -> None:
    """Choose the built UI entry or explicitly announce the API-docs fallback."""
    origin = "http://127.0.0.1:8001"
    assert studio_entry(origin, tmp_path) == (
        "http://127.0.0.1:8001/studios/sc-neurocore/",
        ("SC-NeuroCore Studio starting at http://127.0.0.1:8001/studios/sc-neurocore/",),
    )
    assert studio_entry(origin, None) == (
        "http://127.0.0.1:8001/docs",
        (
            "This installation carries no Studio user interface; its API documentation opens.",
            "SC-NeuroCore Studio starting at http://127.0.0.1:8001/docs",
        ),
    )
    assert studio_entry(origin, None, api_docs_enabled=False) == (
        "http://127.0.0.1:8001/openapi.json",
        (
            "This installation carries no Studio user interface; its OpenAPI schema opens.",
            "SC-NeuroCore Studio starting at http://127.0.0.1:8001/openapi.json",
        ),
    )
    assert studio_entry(origin, tmp_path, api_docs_enabled=False)[0] == (
        "http://127.0.0.1:8001/studios/sc-neurocore/"
    )


def test_development_retains_interactive_docs_with_hashed_bootstrap_scripts() -> None:
    """Serve development docs under their exact-script CSP while lab disables them."""
    application = create_app()
    with TestClient(application, base_url="http://127.0.0.1") as client:
        for route in ("/docs", "/redoc"):
            response = client.get(route)
            assert response.status_code == 200
            assert "'sha256-" in response.headers["content-security-policy"]
            assert "https://cdn.jsdelivr.net" in response.text
        assert client.get("/openapi.json").json() == application.openapi()
