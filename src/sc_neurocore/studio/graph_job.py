# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Durable graph request and worker contract

"""Bind graph jobs to checked scientific inputs and bounded complete results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

from sc_neurocore.studio.evidence_seal import EvidenceSealError, seal_sha256
from sc_neurocore.studio.network_execution import simulate_graph_spec
from sc_neurocore.studio.network_graph_spec import GraphSpec, resolve_graph
from sc_neurocore.studio.platform.jobs_context import StudioJobContext
from sc_neurocore.studio.platform.jobs_models import (
    DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
    StudioJobRefused,
)

GRAPH_JOB_SCHEMA = "studio.graph.job.v1"
GRAPH_JOB_RESULT_SCHEMA = "studio.graph.job.result.v1"
GRAPH_REQUEST_SEED = "graph-request.json"
GRAPH_RESULT_ARTIFACT = "graph/result.json"
GRAPH_RESULT_MAX_BYTES = DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES


class GraphJobRefused(StudioJobRefused):
    """Refuse graph request custody or result size without a partial success."""


@dataclass(frozen=True, slots=True)
class PreparedGraphJob:
    """Resolved asynchronous graph and the exact finite JSON the worker receives."""

    spec: GraphSpec
    request_bytes: bytes
    request_sha256: str
    request_seal_sha256: str

    @property
    def graph_sha256(self) -> str:
        """Return the digest of the effective graph and its asynchronous limits."""
        return str(self.spec.to_public_dict()["graph_sha256"])

    @property
    def submission_sha256(self) -> str:
        """Bind retries to both exact request bytes and the effective work profile."""
        return hashlib.sha256(
            f"{self.request_sha256}:{self.graph_sha256}".encode("ascii")
        ).hexdigest()


def _json_bytes(value: object, *, max_bytes: int) -> bytes:
    """Encode complete finite JSON, stopping before a bounded buffer can overflow."""
    encoded = bytearray()
    encoder = json.JSONEncoder(allow_nan=False, sort_keys=True, separators=(",", ":"))
    try:
        for chunk in encoder.iterencode(value):
            part = chunk.encode("utf-8")
            if len(encoded) + len(part) > max_bytes:
                raise GraphJobRefused("Graph job JSON exceeds its byte limit.")
            encoded.extend(part)
    except (TypeError, ValueError, RecursionError, UnicodeError) as exc:
        if isinstance(exc, GraphJobRefused):
            raise
        raise GraphJobRefused("Graph job requires finite UTF-8 JSON.") from exc
    return bytes(encoded)


def prepare_graph_job(
    graph: object, *, max_request_bytes: int = DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES
) -> PreparedGraphJob:
    """Validate and bind graph input before allocating any durable job resources.

    Parameters
    ----------
    graph : object
        Existing network graph request, including its scientific settings.
    max_request_bytes : int
        Operator-owned seed-input limit, never a graph request field.

    Returns
    -------
    PreparedGraphJob
        Asynchronous specification and bounded canonical request bytes.

    Raises
    ------
    GraphRejected
        The graph cannot execute within the asynchronous work profile.
    GraphJobRefused
        Input JSON cannot be represented portably or exceeds its byte allowance.
    """
    request_bytes = _json_bytes(graph, max_bytes=max_request_bytes)
    spec = resolve_graph(graph, execution_mode="async_job")
    try:
        request_seal = seal_sha256(graph)
    except EvidenceSealError as exc:
        raise GraphJobRefused("Graph job request cannot be sealed portably.") from exc
    return PreparedGraphJob(
        spec, request_bytes, hashlib.sha256(request_bytes).hexdigest(), request_seal
    )


def run_graph_job(context: StudioJobContext, payload: Mapping[str, object]) -> dict[str, object]:
    """Run one digest-bound graph through the maintained public Network runtime.

    Parameters
    ----------
    context : StudioJobContext
        Existing supervisor's confined cancellation, seed and artifact access.
    payload : Mapping[str, object]
        Exact request and graph digests; scientific bytes come from the seed.

    Returns
    -------
    dict[str, object]
        Compact, versioned receipt binding the complete graph-result artifact.
        Scientific arrays remain in that declared, integrity-checked file so
        both execution profiles preserve their bounded control messages.

    Raises
    ------
    GraphJobRefused
        The seed binding or bounded complete-result contract cannot be met.
    GraphRejected
        The stored request no longer meets the graph contract.
    StudioJobCancelled
        Cancellation was requested before execution or result publication.
    """
    context.check_cancelled()
    if set(payload) != {"graph_sha256", "request_sha256"}:
        raise GraphJobRefused("Graph job envelope is invalid.")
    seed = context.read_seed_input(GRAPH_REQUEST_SEED)
    if hashlib.sha256(seed).hexdigest() != payload["request_sha256"]:
        raise GraphJobRefused("Graph job request digest does not match.")
    try:
        graph = json.loads(seed)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise GraphJobRefused("Graph job seed JSON is invalid.") from exc
    prepared = prepare_graph_job(graph)
    if prepared.graph_sha256 != payload["graph_sha256"]:
        raise GraphJobRefused("Graph job specification digest does not match.")
    result = simulate_graph_spec(prepared.spec)
    encoded = _json_bytes(result, max_bytes=GRAPH_RESULT_MAX_BYTES)
    context.check_cancelled()
    context.write_artifact(GRAPH_RESULT_ARTIFACT, encoded)
    return {
        "schema_version": GRAPH_JOB_RESULT_SCHEMA,
        "graph_sha256": prepared.graph_sha256,
        "result_artifact": GRAPH_RESULT_ARTIFACT,
        "result_sha256": hashlib.sha256(encoded).hexdigest(),
        "result_size_bytes": len(encoded),
        "n_steps": result["n_steps"],
        "n_total": result["n_total"],
        "n_spikes": result["n_spikes"],
    }
