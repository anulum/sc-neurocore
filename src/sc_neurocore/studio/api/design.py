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

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from sc_neurocore.studio.api.common import _safe
from sc_neurocore.studio.api.runtime import StudioApiContext
from sc_neurocore.studio.network_execution import GraphExecutionFailure
from sc_neurocore.studio.network_nir import (
    NIRMappingRefused,
    graph_to_nir_file,
    nir_file_to_graph,
)
from sc_neurocore.studio.network_graph import (
    GraphRejected,
    available_models as graph_available_models,
    create_population,
    create_projection,
    graph_issues,
    envelope_to_graph,
    population_model_contract,
    simulate_graph,
)
from sc_neurocore.studio.project import (
    delete_project,
    export_project,
    branch_refused_edit,
    fork_project,
    import_project,
    list_deleted_projects,
    list_projects,
    load_project,
    project_revisions,
    restore_project,
    save_project,
)


class ReviewCommentBody(BaseModel):
    """A review comment on one revision."""

    model_config = ConfigDict(extra="forbid")

    body: str
    reply_to: str | None = None


def build_design_router(context: StudioApiContext) -> APIRouter:
    """Build the project and network-design router over shared Studio runtime state."""
    router = APIRouter()

    @router.post("/api/project/save")
    def api_project_save(data: dict[str, Any]) -> Any:
        """Save Studio state as a new immutable workspace revision.

        ``expected_revision`` is the revision the caller loaded. Omitting it
        claims the workspace is new; saving from a revision that is no longer
        current is refused by the error boundary with HTTP 409 and the revision
        that is current, so a second editor is told rather than overwritten.
        """
        name = data.get("name", "")
        state = data.get("state", {})
        if not name:
            raise HTTPException(422, "Project name required")
        expected = data.get("expected_revision")
        if expected is not None and not isinstance(expected, int):
            raise HTTPException(422, "expected_revision must be an integer revision number")
        return _safe(lambda: save_project(name, state, expected_revision=expected))

    @router.get("/api/project/list")
    def api_project_list() -> Any:
        """List saved workspaces with their current revision and history depth."""
        return list_projects()

    @router.get("/api/project/load/{name}")
    def api_project_load(name: str, revision: int | None = None) -> Any:
        """Load one workspace revision, defaulting to the current one."""
        result = _safe(lambda: load_project(name, revision=revision))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.get("/api/project/{name}/revisions")
    def api_project_revisions(name: str) -> Any:
        """List every stored revision of one workspace, oldest first."""
        return _safe(lambda: {"name": name, "revisions": project_revisions(name)})

    @router.get("/api/project/{name}/comments")
    def api_project_comments(name: str, revision: int | None = None) -> Any:
        """List a workspace's review comments, each checked against its revision."""
        from sc_neurocore.studio.project import review_comments

        try:
            return review_comments(name, revision=revision)
        except KeyError as exc:
            raise HTTPException(404, f"Project '{name}' not found") from exc

    @router.post("/api/project/{name}/revisions/{revision}/comments")
    def api_project_comment(
        name: str, revision: int, comment: ReviewCommentBody, request: Request
    ) -> Any:
        """Comment on one immutable revision; the author is the request's principal."""
        from sc_neurocore.studio.project import comment_on_revision

        principal = getattr(request.state, "studio_principal", None)
        author = principal.principal_id if principal is not None else "local"
        try:
            return comment_on_revision(
                name, revision, author=author, body=comment.body, reply_to=comment.reply_to
            )
        except KeyError as exc:
            raise HTTPException(404, f"Project '{name}' has no revision {revision}") from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/api/project/{name}/fork")
    def api_project_fork(name: str, data: dict[str, Any]) -> Any:
        """Copy one revision of a workspace into a new one, leaving the source alone."""
        new_name = data.get("new_name", "")
        if not new_name:
            raise HTTPException(422, "new_name required")
        revision = data.get("revision")
        if revision is not None and not isinstance(revision, int):
            raise HTTPException(422, "revision must be an integer revision number")
        result = _safe(lambda: fork_project(name, new_name, revision=revision))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.post("/api/project/{name}/branch-refused-edit")
    def api_project_branch_refused_edit(name: str, data: dict[str, Any]) -> Any:
        """Keep an edit that a save conflict refused, as a branch of its own.

        `POST /api/project/save` refuses a save made from a stale revision so it
        cannot overwrite the other editor's work. That leaves the refused edit
        only in the browser that made it. Posting it here stores it as the first
        revision of its own workspace, so both edits survive and can be
        reconciled afterwards rather than one being retyped.
        """
        state = data.get("state")
        if not isinstance(state, dict):
            raise HTTPException(422, "state required")
        base_revision = data.get("base_revision")
        if not isinstance(base_revision, int):
            raise HTTPException(422, "base_revision must be an integer revision number")
        branch_name = data.get("branch_name")
        if branch_name is not None and not isinstance(branch_name, str):
            raise HTTPException(422, "branch_name must be a string")
        result = _safe(
            lambda: branch_refused_edit(
                name, state, base_revision=base_revision, branch_name=branch_name
            )
        )
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.delete("/api/project/{name}")
    def api_project_delete(name: str) -> Any:
        """Move a workspace to the recoverable trash; nothing is erased."""
        result = _safe(lambda: delete_project(name))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.get("/api/project/deleted")
    def api_project_deleted() -> Any:
        """List the workspaces waiting in the recoverable trash."""
        return _safe(lambda: {"deleted": list_deleted_projects()})

    @router.post("/api/project/restore")
    def api_project_restore(data: dict[str, Any]) -> Any:
        """Restore one deleted workspace under its original name."""
        token = data.get("token", "")
        if not token:
            raise HTTPException(422, "token required")
        result = _safe(lambda: restore_project(token))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.get("/api/project/{name}/export")
    def api_project_export(name: str, revision: int | None = None) -> Any:
        """Export one revision as a self-contained transfer document."""
        result = _safe(lambda: export_project(name, revision=revision))
        if "error" in result:
            raise HTTPException(404, result["error"])
        return result

    @router.post("/api/project/import")
    def api_project_import(data: dict[str, Any]) -> Any:
        """Create a workspace from an exported document."""
        name = data.get("name", "")
        document = data.get("document")
        if not name or not isinstance(document, dict):
            raise HTTPException(422, "name and document required")
        result = _safe(lambda: import_project(name, document))
        if "error" in result:
            raise HTTPException(422, result["error"])
        return result

    @router.get("/api/graph/models")
    def api_graph_models() -> Any:
        return _safe(graph_available_models)

    @router.post("/api/graph/population")
    def api_create_population(data: dict[str, Any]) -> Any:
        return _safe(
            lambda: create_population(
                **{
                    k: v
                    for k, v in data.items()
                    if k in ("label", "model", "count", "neuron_type", "x", "y", "params", "drive")
                }
            )
        )

    @router.post("/api/graph/projection")
    def api_create_projection(data: dict[str, Any]) -> Any:
        return _safe(
            lambda: create_projection(
                **{
                    k: v
                    for k, v in data.items()
                    if k in ("source_id", "target_id", "weight", "delay", "probability", "rule")
                }
            )
        )

    @router.get("/api/graph/models/{name}")
    def api_graph_model_contract(name: str) -> Any:
        """Return what a population of one model may be given, and what it may not.

        The canvas has to let a user change a model's parameters, and it cannot
        do that honestly from a list of names: which constructor fields are
        numerically overridable, their kind and default, and the reason each
        other field is not an input are decided here. A browser that guessed
        would be a second implementation of the run contract, free to drift
        from the one that validates the graph.
        """
        return _safe(
            lambda: (
                population_model_contract(name)
                or (_ for _ in ()).throw(
                    HTTPException(404, f"Model '{name}' cannot form a population")
                )
            )
        )

    @router.post("/api/graph/validate")
    def api_validate_graph(data: dict[str, Any]) -> Any:
        """Report every validation failure, each with the field it belongs to.

        ``errors`` is the flat list of messages this route has always
        answered. ``issues`` adds the request field each message came from —
        ``projections[2].delay``, ``populations[0].params.tau`` — because an
        editor that cannot say *which* projection a message is about leaves the
        reader to find it by reading all of them.
        """
        issues = graph_issues(data)
        return {
            "errors": [issue.message for issue in issues],
            "issues": [{"field": issue.field, "message": issue.message} for issue in issues],
            "valid": not issues,
        }

    @router.post("/api/graph/simulate")
    def api_simulate_graph(data: dict[str, Any]) -> Any:
        """Run a graph; validation failures are a 200 with ``success: false``.

        A resolved graph that fails while running (a raising neuron step or a
        non-finite final state) answers 422 with the path-free failure detail.
        """

        def run() -> Any:
            try:
                return simulate_graph(data)
            except GraphExecutionFailure as exc:
                raise HTTPException(status_code=422, detail=exc.to_public_detail()) from None

        return _safe(run)

    @router.post("/api/graph/export-nir")
    def api_export_nir(data: dict[str, Any]) -> Any:
        """Write the graph as a real NIR file, with what NIR does not carry."""

        def run() -> dict[str, object]:
            try:
                return graph_to_nir_file(data).to_public_dict()
            except (GraphRejected, NIRMappingRefused) as exc:
                raise HTTPException(status_code=422, detail={"reason": str(exc)}) from None

        return _safe(run)

    @router.post("/api/graph/import-nir")
    def api_import_nir(data: dict[str, Any]) -> Any:
        """Read a real NIR file, or a Studio graph envelope an earlier export wrote."""

        def run() -> dict[str, Any]:
            try:
                if "content_base64" in data:
                    return nir_file_to_graph(data["content_base64"])
                try:
                    graph = envelope_to_graph(data)
                except ValueError as exc:
                    # The envelope loader's refusals are sentences written for
                    # the reader; the generic handler would reduce them to
                    # "Invalid input".
                    raise NIRMappingRefused(str(exc)) from exc
                return {
                    "graph": graph,
                    "origin": "studio-envelope",
                    "notes": [
                        "read as a Studio graph envelope (JSON), not as NIR; export again to "
                        "obtain a real NIR file"
                    ],
                }
            except (GraphRejected, NIRMappingRefused) as exc:
                raise HTTPException(status_code=422, detail={"reason": str(exc)}) from None

        return _safe(run)

    return router
