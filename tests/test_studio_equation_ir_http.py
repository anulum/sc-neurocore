# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Authored native compiler HTTP contracts

"""The production compiler routes execute real bounded workers and retain receipts."""

from __future__ import annotations

import gc
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from tests.engine_requirement import require_engine

require_engine()

from sc_neurocore_engine.ir import ScGraphBuilder

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.settings import StudioRuntimeSettings


def _source(expression: str, initial: int) -> dict[str, object]:
    """Declare the HTTP counterexample which formerly returned a fixed LIF graph."""
    return {
        "schema_version": "studio.equation-ir-request.v1",
        "model": {
            "equations": [{"name": "v", "expression": expression}],
            "initial_state": [{"name": "v", "value": initial}],
            "dt": {"value": 0.5},
        },
    }


class EquationIRHttpTests(unittest.TestCase):
    """Exercise actual routers, schema admission and child-process compilation."""

    def test_authored_build_semantic_replay_emission_and_input_refusals(self) -> None:
        """Source changes affect IR and semantic outcomes retain exact worker custody."""
        with tempfile.TemporaryDirectory(prefix="studio-equation-http-") as directory:
            settings = StudioRuntimeSettings(
                job_root_path=str(Path(directory) / "jobs"),
                audit_log_path=str(Path(directory) / "audit.jsonl"),
                job_default_timeout_seconds=60.0,
            )
            app = create_app(settings)
            try:
                with TestClient(
                    app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
                ) as client:
                    first_response = client.post("/api/ir/build", json=_source("I", 0))
                    self.assertEqual(first_response.status_code, 200, first_response.text)
                    first = first_response.json()
                    second_response = client.post("/api/ir/build", json=_source("2 * I", 7))
                    self.assertEqual(second_response.status_code, 200, second_response.text)
                    second = second_response.json()
                    self.assertNotEqual(first["ir_text"], second["ir_text"])
                    self.assertNotEqual(first["source_graph_sha256"], second["source_graph_sha256"])
                    self.assertEqual(first["errors"], [])
                    self.assertTrue(first["valid"])
                    self.assertEqual(first["source_replay"], "not_requested")
                    self.assertNotIn("params_q88", first)
                    self.assertNotIn("lif_step", first["ir_text"])
                    receipt = first["studio_job_receipt"]
                    self.assertEqual(receipt["status"], "completed")
                    self.assertEqual(receipt["kind"], "compiler")
                    self.assertEqual(len(receipt["artifacts"]), 2)
                    self.assertNotEqual(receipt["job_id"], second["studio_job_receipt"]["job_id"])
                    for operation in ("verify", "replay", "emit-sv"):
                        with self.subTest(operation=operation):
                            response = client.post(
                                "/api/ir/" + operation, json={"ir_text": first["ir_text"]}
                            )
                            self.assertEqual(response.status_code, 200, response.text)
                            result = response.json()
                            self.assertEqual(
                                result["source_replay"],
                                "not_requested" if operation == "verify" else "passed",
                            )
                            self.assertEqual(result["source_sha256"], first["source_sha256"])
                            self.assertTrue(result["studio_job_receipt"]["job_id"])
                            if operation == "emit-sv":
                                self.assertIn("module sc_equation_neuron", result["systemverilog"])
                    unsupported = client.post(
                        "/api/ir/build", json={**_source("I", 0), "profile": {"data_width": 72}}
                    )
                    self.assertEqual(
                        unsupported.status_code,
                        422,
                        {
                            "response": unsupported.text,
                            "jobs": [
                                (record.status, record.error)
                                for record in app.state.studio_job_manager.list_records()
                            ],
                        },
                    )
                    detail = unsupported.json()["detail"]
                    self.assertEqual(detail["code"], "unsupported_request")
                    self.assertEqual(detail["stage"], "lowering")
                    self.assertEqual(detail["studio_job_receipt"]["status"], "completed")
                    self.assertNotIn(directory, unsupported.text)
                    for route, body in (
                        ("/api/ir/build", {"equations": ["dv/dt = I"], "init": {"v": 7}}),
                        ("/api/ir/build", {**_source("I", 0), "ignored": True}),
                        (
                            "/api/ir/build",
                            {
                                "schema_version": "studio.equation-ir-request.v1",
                                "model": {
                                    "equations": [
                                        {"name": "v", "expression": "-v + w + I"},
                                        {"name": "w", "expression": "-w"},
                                    ],
                                    "initial_state": [{"name": "v", "value": 2}],
                                },
                            },
                        ),
                        ("/api/ir/verify", {"ir_text": ""}),
                        ("/api/ir/replay", {"ir_text": first["ir_text"], "skip_source": True}),
                    ):
                        with self.subTest(route=route, body=body):
                            rejected = client.post(route, json=body)
                            self.assertEqual(rejected.status_code, 422, rejected.text)
                            self.assertNotIn("studio_job_receipt", rejected.text)
                    builder = ScGraphBuilder("raw_http_cell")
                    value = builder.input("i_ext", "fixed<16,8>")
                    builder.output("value", value)
                    raw = builder.build().to_text()
                    for operation in ("replay", "emit-sv"):
                        with self.subTest(unsourced_operation=operation):
                            response = client.post("/api/ir/" + operation, json={"ir_text": raw})
                            self.assertEqual(response.status_code, 422, response.text)
                            self.assertEqual(response.json()["detail"]["code"], "source_required")
                    explicit = client.post(
                        "/api/ir/emit-sv", json={"ir_text": raw, "source_policy": "structural-only"}
                    )
                    self.assertEqual(explicit.status_code, 200, explicit.text)
                    self.assertEqual(explicit.json()["source_replay"], "unavailable")
                    self.assertIn("module raw_http_cell", explicit.json()["systemverilog"])
                    schema = app.openapi()
                    request_schema = schema["paths"]["/api/ir/build"]["post"]["requestBody"][
                        "content"
                    ]["application/json"]["schema"]
                    self.assertEqual(
                        request_schema["$ref"], "#/components/schemas/EquationIRRequest"
                    )
                    self.assertIn("/api/ir/replay", schema["paths"])
            finally:
                app.state.studio_job_manager._ledger.close()
                gc.collect()
