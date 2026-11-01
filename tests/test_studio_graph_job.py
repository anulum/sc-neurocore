# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Graph worker seed and refusal acceptance

"""Actual process workers reject corrupt envelopes and finite-JSON input violations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sc_neurocore.studio.graph_job import (
    GRAPH_REQUEST_SEED,
    GraphJobRefused,
    prepare_graph_job,
)
from sc_neurocore.studio.platform.jobs import StudioJobManager
from tests.studio_seccomp_support import run_child
from tests.test_studio_graph_jobs import graph


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("envelope", "Graph job envelope is invalid."),
        ("request-digest", "Graph job request digest does not match."),
        ("json", "Graph job seed JSON is invalid."),
        ("encoding", "Graph job seed JSON is invalid."),
        ("nonfinite", "Graph job requires finite UTF-8 JSON."),
        ("portability", "Graph job request cannot be sealed portably."),
        ("spec-digest", "Graph job specification digest does not match."),
    ],
)
def test_real_worker_rejects_seed_or_envelope_corruption(
    tmp_path: Path, case: str, message: str
) -> None:
    """Tampered scientific input produces an exact safe failure with no result or artifact."""
    request = {
        "populations": [
            {
                "id": "p",
                "count": 1,
                "neuron_type": "excitatory",
                "model": "SCLapicqueLIFNeuron",
                "drive": {"kind": "constant", "current": 2.0},
            }
        ],
        "projections": [],
        "duration": 2.0,
        "dt": 0.1,
        "seed": 42,
    }
    prepared = prepare_graph_job(request)
    seed = prepared.request_bytes
    if case == "json":
        seed = b"{"
    elif case == "encoding":
        seed = b"\xff"
    elif case == "nonfinite":
        seed = b'{"duration":NaN}'
    elif case == "portability":
        seed = seed.replace(b'"id":"p"', b'"id":"p","label":"\\ud800"', 1)
    payload: dict[str, object] = {
        "graph_sha256": prepared.graph_sha256,
        "request_sha256": hashlib.sha256(seed).hexdigest(),
    }
    if case == "envelope":
        payload["unregistered"] = True
    elif case == "request-digest":
        payload["request_sha256"] = "0" * 64
    elif case == "spec-digest":
        payload["graph_sha256"] = "0" * 64
    manager = StudioJobManager(
        root=tmp_path / "jobs",
        allowed_kinds=frozenset({"analysis"}),
        default_timeout_seconds=30.0,
    )
    record = manager.submit_process_task(
        kind="analysis",
        owner="graph-operator",
        request_id="graph-seed-custody",
        task_path="sc_neurocore.studio.graph_job:run_graph_job",
        payload=payload,
        seed_inputs={GRAPH_REQUEST_SEED: seed},
    )
    final = manager.wait(record.job_id, timeout_seconds=35.0)
    assert final.status == "failed"
    assert final.result is None and final.artifacts == ()
    assert final.to_public_dict()["error"] == message
    assert manager.status().active_count == 0
    assert manager.unreaped_workers == ()


def test_non_json_python_graph_refuses_before_admission() -> None:
    """The public preparation API cannot serialize arbitrary Python objects into a seed."""
    with pytest.raises(GraphJobRefused, match="finite UTF-8 JSON"):
        prepare_graph_job({"populations": {object()}})


def test_public_python_graph_refuses_an_unrecognized_execution_profile() -> None:
    """An actual untyped Python caller cannot choose an unregistered work-limit profile."""
    result = run_child(
        "import json, sys\n"
        "from sc_neurocore.studio.network_graph_spec import GraphRejected, resolve_graph\n"
        "mode = json.loads(sys.argv[1])\n"
        "request = json.loads(sys.argv[2])\n"
        "try:\n"
        "    resolve_graph(request, execution_mode=mode)\n"
        "except GraphRejected as error:\n"
        "    print(json.dumps(error.to_public_detail()))\n",
        arguments=(json.dumps("unregistered-profile"), json.dumps(graph())),
    )
    assert result == {
        "error": "graph_rejected",
        "field": "execution_mode",
        "reason": "Graph execution mode is unsupported",
    }
