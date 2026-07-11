# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio project and network-design routes

"""Persist Studio projects and adapt network-canvas graph operations."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.network_graph import (
    available_models as graph_available_models,
    create_population,
    create_projection,
    graph_to_nir,
    nir_to_graph,
    simulate_graph,
    validate_graph,
)
from sc_neurocore.studio.project import (
    delete_project,
    list_projects,
    load_project,
    save_project,
)


def build_design_router(context: StudioApiContext) -> APIRouter:
    """Build the project and network-design router over shared Studio runtime state."""
    router = APIRouter()

    @router.post("/api/project/save")
    def api_project_save(data: dict[str, Any]) -> Any:
        name = data.get("name", "")
        state = data.get("state", {})
        if not name:
            raise HTTPException(422, "Project name required")
        return _safe(lambda: save_project(name, state))

    @router.get("/api/project/list")
    def api_project_list() -> Any:
        return list_projects()

    @router.get("/api/project/load/{name}")
    def api_project_load(name: str) -> Any:
        result = _safe(lambda: load_project(name))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.delete("/api/project/{name}")
    def api_project_delete(name: str) -> Any:
        result = _safe(lambda: delete_project(name))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.get("/api/graph/models")
    def api_graph_models() -> Any:
        return _safe(graph_available_models)

    @router.post("/api/graph/population")
    def api_create_population(data: dict[str, Any]) -> Any:
        return create_population(
            **{
                k: v
                for k, v in data.items()
                if k in ("label", "model", "count", "neuron_type", "x", "y")
            }
        )

    @router.post("/api/graph/projection")
    def api_create_projection(data: dict[str, Any]) -> Any:
        return _safe(
            lambda: create_projection(
                **{
                    k: v
                    for k, v in data.items()
                    if k in ("source_id", "target_id", "weight", "delay", "probability")
                }
            )
        )

    @router.post("/api/graph/validate")
    def api_validate_graph(data: dict[str, Any]) -> Any:
        errors = validate_graph(data)
        return {"valid": len(errors) == 0, "errors": errors}

    @router.post("/api/graph/simulate")
    def api_simulate_graph(data: dict[str, Any]) -> Any:
        return _safe(lambda: simulate_graph(data))

    @router.post("/api/graph/export-nir")
    def api_export_nir(data: dict[str, Any]) -> Any:
        return _safe(lambda: graph_to_nir(data))

    @router.post("/api/graph/import-nir")
    def api_import_nir(data: dict[str, Any]) -> Any:
        return _safe(lambda: nir_to_graph(data))

    return router
