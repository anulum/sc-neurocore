# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio frontend mounting

"""Mount the built Studio frontend without changing API routing."""

from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def mount_studio_frontend(app: FastAPI, *, app_module_file: str) -> None:
    """Mount the production frontend when its distribution exists.

    Parameters
    ----------
    app:
        FastAPI application receiving the root route and static mount.
    app_module_file:
        Path of the compatibility application module used as the search anchor.
    """
    primary_dist_dir = os.path.join(
        os.path.dirname(app_module_file), "..", "..", "..", "studio", "frontend", "dist"
    )
    dist_dir: str | None = primary_dist_dir
    if not os.path.isdir(primary_dist_dir):
        fallback_dist_dir = os.path.join(
            os.path.dirname(app_module_file),
            "..",
            "..",
            "..",
            "..",
            "studio",
            "frontend",
            "dist",
        )
        dist_dir = fallback_dist_dir if os.path.isdir(fallback_dist_dir) else None

    @app.get("/")
    def serve_index() -> Any:
        if dist_dir is None:
            raise HTTPException(status_code=503, detail="studio_frontend_not_built")
        return FileResponse(os.path.join(dist_dir, "index.html"))

    if dist_dir is not None:
        app.mount("/", StaticFiles(directory=dist_dir), name="static")
