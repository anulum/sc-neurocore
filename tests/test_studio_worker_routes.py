# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio worker-backed route contracts

"""Contract tests for Studio routes that execute through local worker jobs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import StudioRuntimeSettings
from sc_neurocore.studio.platform.jobs import StudioJobManager, StudioJobRecord


def _client_with_job_root(tmp_path: Path) -> tuple[FastAPI, TestClient]:
    """Create a Studio app whose worker artifacts stay under ``tmp_path``."""

    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        job_default_timeout_seconds=10.0,
    )
    app = create_app(settings)
    return app, TestClient(app, base_url="http://127.0.0.1")


def _job_manager(app: FastAPI) -> StudioJobManager:
    """Return the app-local Studio job manager."""

    return cast(StudioJobManager, app.state.studio_job_manager)


def _single_job(app: FastAPI, *, owner: str, kind: str) -> StudioJobRecord:
    """Return the single job matching the expected route owner and kind."""

    matches = [
        record
        for record in _job_manager(app).list_records()
        if record.owner == owner and record.kind == kind
    ]
    assert len(matches) == 1
    record = matches[0]
    assert record.status == "completed"
    return record


def _artifact_json(
    manager: StudioJobManager,
    record: StudioJobRecord,
    relative_path: str,
) -> dict[str, object]:
    """Read one declared worker artifact as JSON through the manager API."""

    payload = manager.read_artifact(record.job_id, relative_path)
    decoded = json.loads(payload.payload.decode("utf-8"))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def test_synthesis_route_records_bounded_worker_job(tmp_path: Path) -> None:
    """Synthesis returns its legacy payload and persists a job result artifact."""

    app, client = _client_with_job_root(tmp_path)

    response = client.post(
        "/api/synth/run",
        json={"verilog": "module test(); endmodule", "target": "ice40"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["target"] == "ice40"
    assert "success" in data
    record = _single_job(app, owner="studio-synthesis", kind="synthesis")
    assert [artifact.relative_path for artifact in record.artifacts] == ["synthesis/result.json"]
    assert _artifact_json(_job_manager(app), record, "synthesis/result.json") == data


def test_multi_target_synthesis_route_records_bounded_worker_job(tmp_path: Path) -> None:
    """Multi-target synthesis remains synchronous while using the job sandbox."""

    app, client = _client_with_job_root(tmp_path)

    response = client.post(
        "/api/synth/multi-target",
        json={"verilog": "module test(); endmodule"},
    )

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"supported", "targets"}
    record = _single_job(app, owner="studio-synthesis", kind="synthesis")
    assert [artifact.relative_path for artifact in record.artifacts] == [
        "synthesis/multi-target-result.json"
    ]
    assert _artifact_json(_job_manager(app), record, "synthesis/multi-target-result.json") == data


def test_pnr_route_records_bounded_worker_job(tmp_path: Path) -> None:
    """PnR records its result artifact even when the external tool is absent."""

    netlist = tmp_path / "design.json"
    netlist.write_text("{}", encoding="utf-8")
    app, client = _client_with_job_root(tmp_path)

    response = client.post(
        "/api/synth/pnr",
        json={"json_path": str(netlist), "target": "ice40"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "success" in data
    record = _single_job(app, owner="studio-pnr", kind="synthesis")
    assert [artifact.relative_path for artifact in record.artifacts] == [
        "synthesis/pnr-result.json"
    ]
    assert _artifact_json(_job_manager(app), record, "synthesis/pnr-result.json") == data


def test_compile_route_records_bounded_worker_job(tmp_path: Path) -> None:
    """Compiler endpoint emits its current payload through a worker artifact."""

    app, client = _client_with_job_root(tmp_path)

    response = client.post(
        "/api/compile",
        json={
            "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
            "threshold": "v > -50",
            "reset": "v = -65",
            "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "module" in data["verilog"]
    assert data["chars"] > 100
    record = _single_job(app, owner="studio-compiler", kind="compiler")
    assert [artifact.relative_path for artifact in record.artifacts] == ["compiler/result.json"]
    assert _artifact_json(_job_manager(app), record, "compiler/result.json") == data


def test_pipeline_route_records_bounded_worker_job(tmp_path: Path) -> None:
    """Pipeline execution records a bounded compiler-family worker result."""

    app, client = _client_with_job_root(tmp_path)

    response = client.post(
        "/api/pipeline/run",
        json={"graph": {"populations": [], "projections": []}, "target": "ice40"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    record = _single_job(app, owner="studio-pipeline", kind="compiler")
    assert [artifact.relative_path for artifact in record.artifacts] == ["pipeline/result.json"]
    assert _artifact_json(_job_manager(app), record, "pipeline/result.json") == data
