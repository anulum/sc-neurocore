# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Compiler Inspector (Block 2)

"""Provide real Studio HTTP fixtures for equation compilation and code generation."""

from __future__ import annotations

import gc
from collections.abc import Iterator

import fastapi
import pytest

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings

LIF_EQ = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "dt": 0.1,
    "duration": 50.0,
    "current": 30.0,
}


LIF_IR_REQUEST: dict[str, object] = {
    "schema_version": "studio.equation-ir-request.v1",
    "model": {
        "equations": [{"name": "v", "expression": "-(v - E_L) / tau_m + I / C"}],
        "parameters": [
            {"name": "E_L", "value": -65.0},
            {"name": "tau_m", "value": 10.0},
            {"name": "C", "value": 1.0},
        ],
        "initial_state": [{"name": "v", "value": -65.0}],
        "threshold": "v > -50",
        "reset": [{"name": "v", "expression": "-65"}],
        "dt": {"value": 0.1},
    },
}


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """Provide isolated real compiler workers and close their durable job ledger.

    Parameters
    ----------
    tmp_path_factory : pytest.TempPathFactory
        Samsung-backed test allocation supplied by the selected test runner.

    Yields
    ------
    TestClient
        Actual Studio application with module-owned jobs and audit records.
    """
    directory = tmp_path_factory.mktemp("studio-compiler")
    app = create_app(
        StudioRuntimeSettings(
            job_root_path=str(directory / "jobs"),
            audit_log_path=str(directory / "audit.jsonl"),
            job_default_timeout_seconds=60.0,
        )
    )
    try:
        with TestClient(
            app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
        ) as client:
            yield client
    finally:
        app.state.studio_job_manager._ledger.close()
        gc.collect()


__all__ = [
    "annotations",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "LIF_EQ",
    "LIF_IR_REQUEST",
    "client",
]
