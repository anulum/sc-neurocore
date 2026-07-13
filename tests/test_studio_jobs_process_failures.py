# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio job sandbox contract tests

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")


import sc_neurocore.studio.platform.jobs as jobs_module
from sc_neurocore.studio.platform import process_worker


def test_studio_process_result_loader_handles_invalid_payloads(tmp_path: Path) -> None:
    missing = jobs_module._load_process_result(tmp_path / "missing.json")
    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("{", encoding="utf-8")
    invalid_json = jobs_module._load_process_result(invalid_json_path)
    invalid_shape_path = tmp_path / "invalid-shape.json"
    invalid_shape_path.write_text("[]", encoding="utf-8")
    invalid_shape = jobs_module._load_process_result(invalid_shape_path)
    malformed_artifacts_path = tmp_path / "malformed-artifacts.json"
    malformed_artifacts_path.write_text(
        json.dumps({"artifacts": [{"relative_path": 1}], "status": "completed"}),
        encoding="utf-8",
    )
    malformed = jobs_module._load_process_result(malformed_artifacts_path)
    not_list_path = tmp_path / "not-list-artifacts.json"
    not_list_path.write_text(
        json.dumps({"artifacts": {}, "status": "completed"}),
        encoding="utf-8",
    )
    non_dict_path = tmp_path / "non-dict-artifact.json"
    non_dict_path.write_text(
        json.dumps({"artifacts": [1], "status": "completed"}),
        encoding="utf-8",
    )
    bad_size_path = tmp_path / "bad-size-artifact.json"
    bad_size_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": "reports/result.txt",
                        "sha256": "0" * 64,
                        "size_bytes": "bad",
                    }
                ],
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    bad_hash_path = tmp_path / "bad-hash-artifact.json"
    bad_hash_path.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "relative_path": "reports/result.txt",
                        "sha256": 1,
                        "size_bytes": 1,
                    }
                ],
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    assert missing.error == "Studio process worker did not write a result."
    assert invalid_json.error == "Studio process worker wrote an invalid result."
    assert invalid_shape.error == "Studio process worker wrote an invalid result."
    assert malformed.artifacts == ()
    assert jobs_module._load_process_artifacts(malformed_artifacts_path) == ()
    assert jobs_module._load_process_result(not_list_path).artifacts == ()
    assert jobs_module._load_process_result(non_dict_path).artifacts == ()
    assert jobs_module._load_process_result(bad_size_path).artifacts == ()
    assert jobs_module._load_process_result(bad_hash_path).artifacts == ()
    assert jobs_module._load_process_artifacts(tmp_path / "missing.json") == ()


def test_studio_process_worker_environment_bootstraps_missing_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Process-worker environment sets PYTHONPATH when it is absent."""

    monkeypatch.delenv("PYTHONPATH", raising=False)

    environment = jobs_module._process_worker_environment()

    assert "PYTHONPATH" in environment
    assert str(Path(jobs_module.__file__).resolve().parents[3]) in environment["PYTHONPATH"]


def test_studio_process_terminate_falls_back_to_kill() -> None:
    class BlockingProcess:
        def __init__(self) -> None:
            self.terminated = False
            self.killed = False
            self.wait_calls = 0

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.wait_calls += 1
            if self.wait_calls == 1:
                raise subprocess.TimeoutExpired(cmd="worker", timeout=1.0)
            return 0

        def kill(self) -> None:
            self.killed = True

    process = BlockingProcess()

    jobs_module._terminate_process(cast(subprocess.Popen[bytes], process))

    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 2


def test_studio_process_worker_main_writes_result_files(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text(json.dumps({"model": "lif"}), encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:process_echo_task",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert payload["status"] == "completed"
    assert payload["result"] == {"payload": {"model": "lif"}, "worker_job_id": "sj_worker"}
    assert payload["artifacts"][0]["relative_path"] == "reports/process-result.txt"


def test_studio_process_worker_main_records_failure(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text("[]", encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:process_echo_task",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "ValueError"


def test_studio_process_worker_main_rejects_non_callable_task(tmp_path: Path) -> None:
    work_dir = tmp_path / "sj_worker"
    work_dir.mkdir()
    payload_path = tmp_path / "payload.json"
    result_path = tmp_path / "result.json"
    payload_path.write_text("{}", encoding="utf-8")

    exit_code = process_worker.main(
        [
            "--task",
            "tests.studio_job_tasks:NON_CALLABLE_TASK",
            "--payload",
            str(payload_path),
            "--result",
            str(result_path),
            "--work-dir",
            str(work_dir),
            "--max-artifact-bytes",
            "1024",
        ]
    )
    payload = json.loads(result_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert payload["status"] == "failed"
    assert payload["error"] == "TypeError"
