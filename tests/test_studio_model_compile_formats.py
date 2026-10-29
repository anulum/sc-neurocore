# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Model compiler format admission before worker execution

"""Reject non-offered model formats through the actual HTTP request boundary."""

from pathlib import Path

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.api.schemas import ModelCompileRequest, ModelCosimRequest
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.model_numeric_contracts import STUDIO_Q_FORMATS
from sc_neurocore.studio.platform import StudioRuntimeSettings


@pytest.mark.parametrize("route", ["/api/models/compile", "/api/models/cosim"])
def test_non_offered_formats_refuse_before_job_admission(tmp_path: Path, route: str) -> None:
    """An unsupported geometry is caller input, never a failed compiler job.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Test-owned jobs and audit location.
    route : str
        Actual compile or inherited co-simulation HTTP request surface.
    """
    settings = StudioRuntimeSettings(
        job_root_path=str(tmp_path / "jobs"),
        audit_log_path=str(tmp_path / "audit.jsonl"),
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1") as client:
        before = client.get("/api/studio/jobs").json()
        for q_format in ["Q32.32", "Q12.4", "Q64.0", "Q0.16", "q8.8", "invalid", 16]:
            response = client.post(
                route,
                json={"model_name": "ExpIFNeuron", "q_format": q_format},
                headers={"Origin": "http://127.0.0.1:5173"},
            )
            assert response.status_code == 422, response.text
            assert client.get("/api/studio/jobs").json() == before


@pytest.mark.parametrize("request_type", [ModelCompileRequest, ModelCosimRequest])
def test_supported_formats_and_existing_numeric_spelling_remain_admitted(
    request_type: type[ModelCompileRequest],
) -> None:
    """Preserve the declared formats, default and previously admitted zero padding.

    Parameters
    ----------
    request_type : type of ModelCompileRequest
        Public request model, including the co-simulation specialization.
    """
    assert request_type(model_name="ExpIFNeuron").q_format == "Q8.8"
    for q_format in [*STUDIO_Q_FORMATS, "Q08.08", "Q016.016"]:
        request = request_type(model_name="ExpIFNeuron", q_format=q_format)
        assert request.model_dump()["q_format"] == q_format
