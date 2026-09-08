# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Synchronous process response identity contracts

"""Bind response receipts to real process jobs without invoking EDA tools."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from sc_neurocore.studio.api.runtime import build_studio_api_context
from sc_neurocore.studio.api import synthesis as synthesis_routes
from sc_neurocore.studio.platform import build_default_studio_runtime_settings


def test_process_receipt_names_exact_job_and_preserves_stored_result(tmp_path: Path) -> None:
    """Two real workers retain distinct identities; envelope does not alter evidence."""
    settings = replace(build_default_studio_runtime_settings(), job_root_path=str(tmp_path))
    context = build_studio_api_context(FastAPI(), settings)
    responses = [
        context.run_studio_process_job_sync(
            kind="synthesis",
            owner="receipt-test",
            task_path="tests.studio_job_tasks:process_echo_task",
            payload={"sequence": sequence},
            include_job_receipt=True,
        )
        for sequence in range(2)
    ]
    for response in responses:
        receipt = response["studio_job_receipt"]
        assert receipt["job_id"] == response["worker_job_id"]
        assert receipt["status"] == "completed"
        assert receipt["schema_version"] == "studio.job-receipt.v1"
        assert receipt["artifacts"] == [
            {
                "relative_path": "reports/process-result.txt",
                "sha256": sha256(b"process ok").hexdigest(),
                "size_bytes": len(b"process ok"),
            }
        ]
        completed = context.studio_job_manager.wait(receipt["job_id"], timeout_seconds=1)
        assert completed.result == {
            key: value for key, value in response.items() if key != "studio_job_receipt"
        }
        assert str(tmp_path) not in str(receipt)
    assert (
        responses[0]["studio_job_receipt"]["job_id"] != responses[1]["studio_job_receipt"]["job_id"]
    )


def test_default_process_response_retains_existing_shape(tmp_path: Path) -> None:
    """Non-opted-in callers retain their existing result contract."""
    settings = replace(build_default_studio_runtime_settings(), job_root_path=str(tmp_path))
    context = build_studio_api_context(FastAPI(), settings)
    result = context.run_studio_process_job_sync(
        kind="compiler",
        owner="receipt-test",
        task_path="tests.studio_job_tasks:process_echo_task",
        payload={"sequence": 1},
    )
    assert set(result) == {"payload", "worker_job_id"}


@pytest.mark.parametrize(
    ("route", "task_constant", "payload"),
    [
        ("run", "SYNTHESIS_RUN_PROCESS_TASK", {"verilog": "module t; endmodule"}),
        ("multi-target", "SYNTHESIS_MULTI_TARGET_PROCESS_TASK", {"verilog": "module t; endmodule"}),
        (
            "terminal",
            "SYNTHESIS_TERMINAL_PROCESS_TASK",
            {
                "verilog": "module t; endmodule",
                "target": "ice40",
                "compile_traceability": {},
                "cosim_parity": {},
            },
        ),
        ("pnr", "SYNTHESIS_PNR_PROCESS_TASK", {"json_path": "controlled.json"}),
    ],
)
def test_synthesis_routes_return_the_executed_process_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    task_constant: str,
    payload: dict[str, object],
) -> None:
    """Real routing and process execution preserve identity; substitute EDA task only."""
    monkeypatch.setattr(synthesis_routes, task_constant, "tests.studio_job_tasks:process_echo_task")
    app = FastAPI()
    context = build_studio_api_context(
        app,
        replace(build_default_studio_runtime_settings(), job_root_path=str(tmp_path)),
    )
    app.include_router(synthesis_routes.build_synthesis_router(context))
    with TestClient(app) as client:
        response = client.post(f"/api/synth/{route}", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["studio_job_receipt"]["job_id"] == result["worker_job_id"]
    assert result["studio_job_receipt"]["kind"] == "synthesis"
    assert (
        result["studio_job_receipt"]["artifacts"][0]["relative_path"]
        == "reports/process-result.txt"
    )
