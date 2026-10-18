# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio frontend mounting

"""Serve the built Studio frontend beside the API, from the package or a checkout.

The frontend is built for the path it is published under,
:data:`STUDIO_UI_PATH`, and every asset it loads names that path, so it is
mounted there; the root redirects to it. An installed distribution carries its
own build in ``sc_neurocore/studio/frontend_dist``; a checkout serves
``studio/frontend/dist`` once it has been built. Without either there is no
user interface to serve, and the root is left unclaimed.
"""

from __future__ import annotations

import base64
import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from sc_neurocore.studio.platform.settings import DEFAULT_STUDIO_HTTP_SECURITY_HEADERS

#: Where the built frontend expects to be served; its assets name this path.
STUDIO_UI_PATH = "/studios/sc-neurocore"

_PACKAGED_UI = Path(__file__).resolve().parents[1] / "frontend_dist"


def _docs_response(response: HTMLResponse, headers: Mapping[str, str]) -> HTMLResponse:
    """Allow only the generated documentation bootstrap script and its documentation CDN."""
    hashes = []
    for match in re.finditer(
        r"<script(?:\s[^>]*)?>(.*?)</script>", bytes(response.body).decode(), re.DOTALL
    ):
        digest = base64.b64encode(hashlib.sha256(match.group(1).encode()).digest()).decode("ascii")
        hashes.append(f"'sha256-{digest}'")
    policy = headers.get(
        "content-security-policy", DEFAULT_STUDIO_HTTP_SECURITY_HEADERS["content-security-policy"]
    )
    policy = policy.replace(
        "script-src 'self'", "script-src 'self' https://cdn.jsdelivr.net " + " ".join(hashes)
    )
    policy = policy.replace("style-src 'self'", "style-src 'self' https://cdn.jsdelivr.net")
    response.headers["content-security-policy"] = policy
    return response


def mount_studio_api_docs(app: FastAPI, *, security_headers: Mapping[str, str]) -> None:
    """Serve interactive API docs under a CSP with exact bootstrap-script hashes."""

    @app.get("/docs", include_in_schema=False)
    def api_docs() -> HTMLResponse:
        """Serve Swagger without allowing arbitrary inline JavaScript."""
        return _docs_response(
            get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} API"),
            security_headers,
        )

    @app.get("/redoc", include_in_schema=False)
    def api_redoc() -> HTMLResponse:
        """Serve ReDoc using the same bounded documentation script policy."""
        return _docs_response(
            get_redoc_html(
                openapi_url="/openapi.json", title=f"{app.title} API", with_google_fonts=False
            ),
            security_headers,
        )


def studio_frontend_candidates(app_module_file: str) -> tuple[Path, ...]:
    """Return where a built frontend is looked for, in order.

    The packaged build comes first, so an installation serves its own; the
    checkout's build follows, found from the application module.
    """
    anchor = Path(app_module_file).resolve().parent
    return (
        _PACKAGED_UI,
        anchor.parents[2] / "studio" / "frontend" / "dist",
        anchor.parents[3] / "studio" / "frontend" / "dist",
    )


def studio_frontend_dir(candidates: Sequence[Path]) -> Path | None:
    """Return the first candidate holding a built frontend, or ``None``."""
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


def studio_entry(origin: str, dist_dir: Path | None) -> tuple[str, tuple[str, ...]]:
    """Return the page a launched Studio opens and the lines announcing it.

    Parameters
    ----------
    origin:
        Scheme, host and port the API serves, without a trailing slash.
    dist_dir:
        The built frontend :func:`studio_frontend_dir` found, or ``None``.

    Returns
    -------
    tuple
        The URL to open, and the lines to print before opening it. Without a
        frontend the root would be an empty page, so the API documentation
        opens instead and the first line says why.
    """
    if dist_dir is None:
        url = f"{origin}/docs"
        return url, (
            "This installation carries no Studio user interface; its API documentation opens.",
            f"SC-NeuroCore Studio starting at {url}",
        )
    url = f"{origin}{STUDIO_UI_PATH}/"
    return url, (f"SC-NeuroCore Studio starting at {url}",)


def mount_studio_frontend(app: FastAPI, *, app_module_file: str) -> None:
    """Mount the built frontend at :data:`STUDIO_UI_PATH` when one exists.

    Parameters
    ----------
    app:
        FastAPI application receiving the root redirect and the static mount.
    app_module_file:
        Path of the application module, the anchor for the checkout's build.
    """
    dist_dir = studio_frontend_dir(studio_frontend_candidates(app_module_file))
    if dist_dir is None:
        return

    @app.get("/", include_in_schema=False)
    def serve_index() -> Any:
        return RedirectResponse(f"{STUDIO_UI_PATH}/")

    app.mount(STUDIO_UI_PATH, StaticFiles(directory=dist_dir, html=True), name="studio-ui")


__all__ = [
    "STUDIO_UI_PATH",
    "mount_studio_frontend",
    "mount_studio_api_docs",
    "studio_frontend_candidates",
    "studio_entry",
    "studio_frontend_dir",
]
