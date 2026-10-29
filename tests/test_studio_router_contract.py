# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio responsibility-router contract tests

"""Verify Studio route ownership and the committed OpenAPI contract."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

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
        "sc_neurocore.studio.api.host_jobs",
        "sc_neurocore.studio.api.hardware",
        "sc_neurocore.studio.api.presets",
        "sc_neurocore.studio.api.simulation",
        "sc_neurocore.studio.api.synthesis",
        "sc_neurocore.studio.api.system",
        "sc_neurocore.studio.api.training",
        "sc_neurocore.studio.api.training_weights",
    }
)


class _DocumentationMarkup(HTMLParser):
    """Read resource URLs and literal script slices from an actual HTTP document."""

    def __init__(self, document: str) -> None:
        super().__init__()
        self.resources: list[str] = []
        self.images: list[str] = []
        self.inline_scripts: list[str] = []
        self._document = document
        self._lines = document.splitlines(keepends=True)
        self._start: int | None = None
        self.feed(document)
        self.close()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Locate generated script bodies and external documentation resources.

        Parameters
        ----------
        tag : str
            Parsed element name from the response.
        attrs : list of tuple of str and str or None
            Parsed attributes from that element.
        """
        attributes = dict(attrs)
        if tag == "script":
            if "src" in attributes:
                source = attributes["src"]
                assert source is not None
                self.resources.append(source)
            else:
                line, column = self.getpos()
                start_tag = self.get_starttag_text()
                assert start_tag is not None
                self._start = sum(map(len, self._lines[: line - 1])) + column + len(start_tag)
        elif tag == "link" and attributes.get("rel") == "stylesheet":
            source = attributes.get("href")
            assert source is not None
            self.resources.append(source)
        elif tag == "link" and attributes.get("rel") == "shortcut icon":
            source = attributes.get("href")
            assert source is not None
            self.images.append(source)

    def handle_endtag(self, tag: str) -> None:
        """Record the exact response slice between an inline script's tags.

        Parameters
        ----------
        tag : str
            Parsed closing element name.
        """
        if tag == "script" and self._start is not None:
            line, column = self.getpos()
            end = sum(map(len, self._lines[: line - 1])) + column
            self.inline_scripts.append(self._document[self._start : end])
            self._start = None


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
    # The authored compiler added POST /api/ir/replay, bringing the baseline to151.
    # The three closed hosted description/submission/cancellation routes add three.
    # Read-only hardware registration discovery adds one public route.
    assert len(backend_routes) == 155
    assert {
        (route.path, tuple(sorted(route.methods or ())))
        for route in backend_routes
        if route.endpoint.__module__ == "sc_neurocore.studio.api.host_jobs"
    } == {
        ("/api/studio/host-bridge", ("GET",)),
        ("/api/studio/host-jobs", ("POST",)),
        ("/api/studio/jobs/{job_id}/cancel", ("POST",)),
    }
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
    """Bind actual development docs scripts and CDN resources to their response policy."""
    application = create_app()
    with TestClient(application, base_url="http://127.0.0.1") as client:
        for route in ("/docs", "/redoc"):
            response = client.get(route)
            assert response.status_code == 200
            markup = _DocumentationMarkup(response.text)
            assert markup.resources
            for source in markup.resources:
                resource = urlsplit(source)
                assert resource.scheme == "https"
                assert resource.netloc == "cdn.jsdelivr.net"
                assert resource.path.startswith("/npm/")
            hashes = {
                "'sha256-"
                + base64.b64encode(hashlib.sha256(script.encode()).digest()).decode("ascii")
                + "'"
                for script in markup.inline_scripts
            }
            directives = {
                values[0]: set(values[1:])
                for part in response.headers["content-security-policy"].split(";")
                if (values := part.split())
            }
            assert directives["script-src"] == {"'self'", "https://cdn.jsdelivr.net", *hashes}
            assert directives["style-src"] == {
                "'self'",
                "'unsafe-inline'",
                "https://cdn.jsdelivr.net",
            }
            favicon = "https://fastapi.tiangolo.com/img/favicon.png"
            assert markup.images == [favicon]
            image_sources = {favicon}
            if route == "/redoc":
                image_sources.add("https://cdn.redoc.ly/redoc/logo-mini.svg")
            assert directives["img-src"] == {"'self'", "data:", "blob:", *image_sources}
            if route == "/docs":
                assert len(markup.inline_scripts) == 1
                assert "SwaggerUIBundle" in markup.inline_scripts[0]
                assert "/openapi.json" in markup.inline_scripts[0]
        schema_response = client.get("/openapi.json")
        assert schema_response.json() == application.openapi()
        api_directives = {
            values[0]: set(values[1:])
            for part in schema_response.headers["content-security-policy"].split(";")
            if (values := part.split())
        }
        assert api_directives["script-src"] == {"'self'"}
        assert api_directives["img-src"] == {"'self'", "data:", "blob:"}
