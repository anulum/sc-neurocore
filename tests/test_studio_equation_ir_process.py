# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Native compiler process-boundary contracts

"""Real process jobs preserve authored requests and publish truthful outcome evidence."""

from __future__ import annotations

import gc
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError
from sc_neurocore_engine.ir import parse_ir

from sc_neurocore.studio.equation_ir import build_equation_ir
from sc_neurocore.studio.equation_ir_schema import EquationIRRequest
from sc_neurocore.studio.platform.equation_ir_process import (
    EQUATION_IR_PROCESS_TASK,
    EquationIRWorkerResult,
)
from sc_neurocore.studio.platform.jobs import StudioJobManager, StudioJobRecord


class EquationIRProcessTests(unittest.TestCase):
    """Exercise serialization, native execution, job custody and artifact reads."""

    def test_process_build_replay_emission_and_refusal_evidence(self) -> None:
        """Actual child processes preserve order/keys and separate refusal from completion."""
        request = EquationIRRequest.model_validate(
            {
                "schema_version": "studio.equation-ir-request.v1",
                "model": {
                    "equations": [
                        {"name": "z", "expression": "I - a"},
                        {"name": "a", "expression": "z - a"},
                    ],
                    "initial_state": [{"name": "z", "value": 7}, {"name": "a", "value": 2}],
                    "method": "gauss_seidel",
                    "rng_seed": str(2**64 - 1),
                    "random_stream": 2**32 - 1,
                },
                "profile": {"data_width": 32, "fraction": 16},
            }
        )
        expected = build_equation_ir(request)
        with tempfile.TemporaryDirectory(prefix="studio-equation-ir-") as directory:
            manager = StudioJobManager(
                root=Path(directory),
                allowed_kinds=frozenset({"compiler"}),
                default_timeout_seconds=60.0,
            )
            try:
                built = self._run(manager, "build", request.model_dump(mode="json"))
                envelope = EquationIRWorkerResult.model_validate(built.result).root
                self.assertTrue(envelope.accepted)
                self.assertIsNotNone(envelope.result)
                actual = envelope.result
                if actual is None:
                    self.fail("Accepted native build has no result")
                self.assertEqual(actual.ir_text, expected.ir_text)
                source = json.loads(str(parse_ir(actual.ir_text).source_payload))
                self.assertEqual([pair[0] for pair in source["model"]["equations"]], ["z", "a"])
                self.assertEqual(source["model"]["rng_seed"], 2**64 - 1)
                self.assertEqual(source["model"]["random_stream"], 2**32 - 1)
                for operation in ("verify", "replay", "emit-sv"):
                    with self.subTest(operation=operation):
                        record = self._run(manager, operation, {"ir_text": actual.ir_text})
                        result = EquationIRWorkerResult.model_validate(record.result).root
                        self.assertTrue(result.accepted)
                        if result.result is None:
                            self.fail("Accepted native operation has no result")
                        self.assertEqual(
                            result.result.source_replay,
                            "not_requested" if operation == "verify" else "passed",
                        )
                        if operation == "emit-sv":
                            self.assertIn(
                                "module sc_equation_neuron", result.result.systemverilog or ""
                            )
                oversized = request.model_dump(mode="json")
                oversized["profile"]["data_width"] = 72
                refused = self._run(manager, "build", oversized)
                refusal = EquationIRWorkerResult.model_validate(refused.result).root
                self.assertFalse(refusal.accepted)
                self.assertIsNone(refusal.result)
                if refusal.refusal is None:
                    self.fail("Refused compilation has no reason")
                self.assertEqual(refusal.refusal.code, "unsupported_request")
                self.assertEqual(refusal.refusal.status_code, 422)
                evidence = json.loads(
                    manager.read_artifact(refused.job_id, "equation-ir/evidence.json").payload
                )
                self.assertEqual(evidence["status"], "failed")
                self.assertEqual(refused.status, "completed")
                for contradictory in (
                    {**refusal.model_dump(), "accepted": True},
                    {**refusal.model_dump(), "result": actual.model_dump()},
                    {
                        **refusal.model_dump(),
                        "refusal": {**refusal.refusal.model_dump(), "status_code": 503},
                    },
                ):
                    with self.subTest(envelope=contradictory), self.assertRaises(ValidationError):
                        EquationIRWorkerResult.model_validate(contradictory)
                invalid = self._run(manager, "build", {"equations": ["dv/dt = I"]})
                invalid_result = EquationIRWorkerResult.model_validate(invalid.result).root
                if invalid_result.refusal is None:
                    self.fail("Malformed worker request was accepted")
                self.assertEqual(invalid_result.refusal.stage, "request")
                self.assertEqual(invalid_result.refusal.code, "invalid_input")
                unknown = manager.submit_process_task(
                    kind="compiler",
                    owner="studio-equation-ir",
                    request_id=None,
                    task_path=EQUATION_IR_PROCESS_TASK,
                    payload={"operation": "unregistered", "request": {}},
                )
                rejected = manager.wait(unknown.job_id, 65.0)
                self.assertEqual(rejected.status, "failed")
                self.assertIsNone(rejected.result)
                self.assertFalse(
                    any(
                        item.relative_path == "equation-ir/evidence.json"
                        for item in rejected.artifacts
                    )
                )
                self.assertEqual(manager.unreaped_workers, ())
            finally:
                for record in manager.list_records():
                    if record.status not in {"completed", "failed", "cancelled", "timed_out"}:
                        manager.cancel(record.job_id)
                        manager.wait(record.job_id, 15.0)
                manager._ledger.close()
                gc.collect()

    def _run(
        self, manager: StudioJobManager, operation: str, request: dict[str, object]
    ) -> StudioJobRecord:
        """Submit an actual production process task and verify both output artifacts."""
        submitted = manager.submit_process_task(
            kind="compiler",
            owner="studio-equation-ir",
            request_id=None,
            task_path=EQUATION_IR_PROCESS_TASK,
            payload={"operation": operation, "request": request},
        )
        completed = manager.wait(submitted.job_id, 65.0)
        self.assertEqual(completed.status, "completed", completed.error)
        self.assertEqual(completed.execution_model, "process")
        self.assertEqual(completed.owner, "studio-equation-ir")
        self.assertEqual(len(completed.artifacts), 2)
        payload = manager.read_artifact(completed.job_id, "equation-ir/result.json")
        self.assertEqual(hashlib.sha256(payload.payload).hexdigest(), payload.artifact.sha256)
        self.assertEqual(json.loads(payload.payload), completed.result)
        evidence = json.loads(
            manager.read_artifact(completed.job_id, "equation-ir/evidence.json").payload
        )
        self.assertEqual(evidence["job_id"], completed.job_id)
        self.assertEqual(evidence["action_kind"], "studio.equation_ir")
        self.assertEqual(evidence["artifacts"][0]["sha256"], payload.artifact.sha256)
        self.assertEqual(evidence["replay_route"], "POST /api/ir/" + operation)
        self.assertNotIn(str(manager.root), str(completed.to_public_dict()))
        return completed
