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
from collections.abc import Mapping, Sequence
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from sc_neurocore.studio.platform.settings import DEFAULT_STUDIO_HTTP_SECURITY_HEADERS
from sc_neurocore.studio.frontend_resources import (
    studio_frontend_candidates as studio_frontend_candidates,
    studio_frontend_dir as studio_frontend_dir,
)

#: Where the built frontend expects to be served; its assets name this path.
STUDIO_UI_PATH = "/studios/sc-neurocore"

_DOCS_FAVICON = "https://fastapi.tiangolo.com/img/favicon.png"
_REDOC_LOGO = "https://cdn.redoc.ly/redoc/logo-mini.svg"


class _DocumentationScripts(HTMLParser):
    """Collect exact inline script hashes from generated API documentation HTML."""

    def __init__(self) -> None:
        super().__init__()
        self.hashes: list[str] = []
        self._script: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        """Start collecting an inline script element.

        Parameters
        ----------
        tag : str
            Parsed lowercase element name.
        attrs : list of tuple of str and str or None
            Element attributes; a source attribute selects an external script.
        """
        if tag == "script":
            self._script = None if any(name == "src" for name, _ in attrs) else []

    def handle_data(self, data: str) -> None:
        """Retain literal script text, including whitespace and character references.

        Parameters
        ----------
        data : str
            Parsed character data. HTMLParser preserves script raw text.
        """
        if self._script is not None:
            self._script.append(data)

    def handle_endtag(self, tag: str) -> None:
        """Hash a complete inline script and stop collecting its text.

        Parameters
        ----------
        tag : str
            Parsed lowercase closing element name.
        """
        if tag == "script" and self._script is not None:
            digest = hashlib.sha256("".join(self._script).encode()).digest()
            self.hashes.append("'sha256-" + base64.b64encode(digest).decode("ascii") + "'")
            self._script = None


def _docs_response(
    response: HTMLResponse, headers: Mapping[str, str], *, image_sources: Sequence[str]
) -> HTMLResponse:
    """Admit generated bootstrap scripts and the documentation's declared resources."""
    scripts = _DocumentationScripts()
    scripts.feed(bytes(response.body).decode())
    scripts.close()
    policy = headers.get(
        "content-security-policy", DEFAULT_STUDIO_HTTP_SECURITY_HEADERS["content-security-policy"]
    )
    policy = policy.replace(
        "script-src 'self'",
        "script-src 'self' https://cdn.jsdelivr.net " + " ".join(scripts.hashes),
    )
    policy = policy.replace("style-src 'self'", "style-src 'self' https://cdn.jsdelivr.net")
    policy = policy.replace("img-src 'self'", "img-src 'self' " + " ".join(image_sources))
    response.headers["content-security-policy"] = policy
    return response


def mount_studio_api_docs(app: FastAPI, *, security_headers: Mapping[str, str]) -> None:
    """Serve interactive API docs under a CSP with exact bootstrap-script hashes."""

    @app.get("/docs", include_in_schema=False)
    def api_docs() -> HTMLResponse:
        """Serve Swagger without allowing arbitrary inline JavaScript."""
        return _docs_response(
            get_swagger_ui_html(
                openapi_url="/openapi.json",
                title=f"{app.title} API",
                swagger_favicon_url=_DOCS_FAVICON,
            ),
            security_headers,
            image_sources=(_DOCS_FAVICON,),
        )

    @app.get("/redoc", include_in_schema=False)
    def api_redoc() -> HTMLResponse:
        """Serve ReDoc using the same bounded documentation script policy."""
        return _docs_response(
            get_redoc_html(
                openapi_url="/openapi.json",
                title=f"{app.title} API",
                redoc_favicon_url=_DOCS_FAVICON,
                with_google_fonts=False,
            ),
            security_headers,
            image_sources=(_DOCS_FAVICON, _REDOC_LOGO),
        )


def studio_entry(
    origin: str, dist_dir: Path | None, *, api_docs_enabled: bool = True
) -> tuple[str, tuple[str, ...]]:
    """Return the page a launched Studio opens and the lines announcing it.

    Parameters
    ----------
    origin:
        Scheme, host and port the API serves, without a trailing slash.
    dist_dir:
        The built frontend :func:`studio_frontend_dir` found, or ``None``.
    api_docs_enabled:
        Whether interactive documentation is mounted for this runtime profile.

    Returns
    -------
    tuple
        The URL to open, and the lines to print before opening it. Without a
        frontend the API documentation opens in development, or the OpenAPI
        schema when interactive documentation is disabled.
    """
    if dist_dir is None:
        url = f"{origin}/docs" if api_docs_enabled else f"{origin}/openapi.json"
        fallback = "API documentation" if api_docs_enabled else "OpenAPI schema"
        return url, (
            f"This installation carries no Studio user interface; its {fallback} opens.",
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
