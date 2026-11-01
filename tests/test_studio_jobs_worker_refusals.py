# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Finite worker refusal vocabulary contracts

"""Worker diagnostics never become public messages by matching untrusted text alone."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.graph_job import GraphJobRefused
from sc_neurocore.studio.platform.jobs_failures import GENERIC_JOB_FAILURE, public_job_error
from sc_neurocore.studio.platform.jobs_worker_refusals import (
    WORKER_FAILURE_SCHEMA,
    worker_job_error,
    worker_refusal_code,
)


@pytest.mark.parametrize(
    ("code", "message"),
    [
        ("graph_json_size", "Graph job JSON exceeds its byte limit."),
        ("graph_json", "Graph job requires finite UTF-8 JSON."),
        ("graph_envelope", "Graph job envelope is invalid."),
        ("graph_request_digest", "Graph job request digest does not match."),
        ("graph_seed_json", "Graph job seed JSON is invalid."),
        ("graph_spec_digest", "Graph job specification digest does not match."),
    ],
)
def test_graph_codes_render_only_fixed_source_owned_messages(code: str, message: str) -> None:
    """Typed worker refusals round-trip while private diagnostics and ordinary errors stay private."""
    assert worker_refusal_code(GraphJobRefused(message)) == code
    assert worker_refusal_code(ValueError(message)) is None
    diagnostic = "/operator-private/example fault"
    projected = worker_job_error(
        {"failure_schema": WORKER_FAILURE_SCHEMA, "refusal_code": code, "public_error": diagnostic},
        diagnostic=diagnostic,
    )
    assert str(projected) == diagnostic
    assert public_job_error(projected) == message


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"failure_schema": "old", "refusal_code": "graph_json_size"},
        {"failure_schema": WORKER_FAILURE_SCHEMA, "refusal_code": "unregistered"},
        {"failure_schema": WORKER_FAILURE_SCHEMA, "refusal_code": 1},
    ],
)
def test_unknown_or_unversioned_worker_codes_keep_private_diagnostics(
    payload: dict[object, object],
) -> None:
    """An untrusted code, schema or claimed public field cannot expose a worker fault."""
    diagnostic = "/operator-private/example fault"
    assert public_job_error(worker_job_error(payload, diagnostic=diagnostic)) == GENERIC_JOB_FAILURE
    assert worker_refusal_code(GraphJobRefused(diagnostic)) is None
