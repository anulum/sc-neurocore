# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compiler response boundary with actual worker bytes

"""Decode real persisted worker outcomes and controlled corruptions of their wire data."""

from __future__ import annotations

import gc
import json
import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

from sc_neurocore.studio.api.compiler_result import decode_equation_ir_response
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.equation_ir_process import EquationIRWorkerResult
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


class CompilerResponseTests(unittest.TestCase):
    """Use native HTTP jobs and their public artifact reads as decoder input."""

    def test_persisted_outcomes_and_malformed_domain_envelopes(self) -> None:
        """The production decoder retains receipts and sanitizes malformed result bytes."""
        request = {
            "schema_version": "studio.equation-ir-request.v1",
            "model": {"equations": [{"name": "v", "expression": "I"}]},
        }
        with tempfile.TemporaryDirectory(prefix="studio-compiler-result-") as directory:
            app = create_app(
                StudioRuntimeSettings(
                    job_root_path=str(Path(directory) / "jobs"),
                    audit_log_path=str(Path(directory) / "audit.jsonl"),
                    job_default_timeout_seconds=60.0,
                )
            )
            manager = app.state.studio_job_manager
            try:
                with TestClient(
                    app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
                ) as client:
                    built = client.post("/api/ir/build", json=request)
                    self.assertEqual(built.status_code, 200, built.text)
                    receipt = built.json()["studio_job_receipt"]
                    artifact = manager.read_artifact(receipt["job_id"], "equation-ir/result.json")
                    persisted = json.loads(artifact.payload)
                    decoded = decode_equation_ir_response(
                        {**persisted, "studio_job_receipt": receipt}
                    )
                    self.assertEqual(decoded, built.json())
                    self.assertTrue(
                        EquationIRWorkerResult.model_validate_json(artifact.payload).root.accepted
                    )
                    self.assertEqual(
                        decode_equation_ir_response(persisted),
                        {**built.json(), "studio_job_receipt": None},
                    )
                    corruptions = (
                        {**persisted, "accepted": 1},
                        {**persisted, "accepted": 0},
                        {**persisted, "result": None},
                        {**persisted, "accepted": False},
                        {**persisted, "refusal": {"message": directory}},
                        {**persisted, "result": {"local_path": directory}},
                        {key: value for key, value in persisted.items() if key != "accepted"},
                        {**persisted, "unexpected": directory},
                    )
                    for corrupted in corruptions:
                        wire = json.dumps(corrupted, allow_nan=False).encode("utf-8")
                        with self.subTest(wire=wire), self.assertRaises(HTTPException) as caught:
                            decode_equation_ir_response(
                                {
                                    **json.loads(wire),
                                    "studio_job_receipt": receipt,
                                }
                            )
                        self.assertEqual(caught.exception.status_code, 500)
                        self.assertEqual(caught.exception.detail, "studio_compiler_result_invalid")
                        self.assertNotIn(directory, str(caught.exception.detail))
                        self.assertIsInstance(caught.exception.__cause__, ValidationError)
                    with self.assertRaises(ValidationError):
                        EquationIRWorkerResult.model_validate_json(b"[]")
                    refused = client.post(
                        "/api/ir/build", json={**request, "profile": {"data_width": 72}}
                    )
                    self.assertEqual(refused.status_code, 422, refused.text)
                    detail = refused.json()["detail"]
                    refusal_receipt = detail["studio_job_receipt"]
                    refusal_bytes = manager.read_artifact(
                        refusal_receipt["job_id"], "equation-ir/result.json"
                    ).payload
                    with self.assertRaises(HTTPException) as refused_decode:
                        decode_equation_ir_response(
                            {
                                **json.loads(refusal_bytes),
                                "studio_job_receipt": refusal_receipt,
                            }
                        )
                    self.assertEqual(refused_decode.exception.status_code, 422)
                    self.assertEqual(refused_decode.exception.detail, detail)
                    self.assertEqual(manager.unreaped_workers, ())
                    self.assertEqual(
                        {record.status for record in manager.list_records()}, {"completed"}
                    )
            finally:
                manager._ledger.close()
                gc.collect()
