# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — The train command in the measured process

"""``sc-neurocore train`` exit statuses, run inside the test process.

``tests/test_cli_train.py`` runs the console module as a user would, in its
own process, where coverage does not see the command body. These cases call
the same public entry point in this process against a real job root: the
request still goes through the Studio contract, the job manager and a real
worker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sc_neurocore.cli.commands.train import EXIT_CRITERION_MISSED
from tests.cli_test_support import run_cli

torch = pytest.importorskip("torch")

#: A small conversion run on the synthetic dataset; it finishes in seconds.
_RUN: dict[str, object] = {
    "model_kind": "qcfs_conversion",
    "dataset": "synthetic",
    "epochs": 1,
    "batch_size": 32,
    "timesteps": 4,
    "hidden": [16],
    "lr": 0.01,
    "seed": 11,
}


def _request(tmp_path: Path, request: dict[str, object]) -> Path:
    """Write ``request`` as the JSON file the command reads."""
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    return path


def _outcome(capsys: pytest.CaptureFixture[str]) -> dict[str, object]:
    """Return the JSON object the command printed as its last output line."""
    printed = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert isinstance(printed, dict)
    return printed


def test_a_completed_run_reports_its_metrics_and_exits_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run with no criterion completes, prints its sealed outcome and exits 0."""
    root = tmp_path / "jobs"
    status = run_cli("train", str(_request(tmp_path, _RUN)), "--job-root", str(root))
    outcome = _outcome(capsys)
    assert status == 0
    assert outcome["status"] == "completed" and outcome["error"] is None
    assert outcome["preregistration_verdict"] is None
    assert isinstance(outcome["final_metrics"], dict)
    assert outcome["job_root"] == str(root.resolve())
    assert (root / str(outcome["job_id"])).is_dir()


def test_a_missed_criterion_has_its_own_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A completed run whose preregistered criterion is missed exits 3, not 0."""
    request = {**_RUN, "preregistration": {"metric": "val_accuracy", "threshold": 1.0}}
    status = run_cli(
        "train",
        str(_request(tmp_path, request)),
        "--job-root",
        str(tmp_path / "jobs"),
        "--timeout",
        "120",
    )
    outcome = _outcome(capsys)
    assert status == EXIT_CRITERION_MISSED == 3
    assert outcome["status"] == "completed"
    verdict = outcome["preregistration_verdict"]
    assert isinstance(verdict, dict) and verdict["passed"] is False


def test_a_failed_run_exits_one_with_its_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A run that fails in its worker exits 1 and prints the owned refusal."""
    request = {**_RUN, "batch_size": 200}
    status = run_cli(
        "train", str(_request(tmp_path, request)), "--job-root", str(tmp_path / "jobs")
    )
    outcome = _outcome(capsys)
    assert status == 1
    assert outcome["status"] == "failed"
    assert "validation split served no samples" in str(outcome["error"])


def test_a_request_the_contract_refuses_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A request the training contract refuses is reported by field and starts no job."""
    root = tmp_path / "jobs"
    status = run_cli(
        "train", str(_request(tmp_path, {**_RUN, "model_kind": "ann"})), "--job-root", str(root)
    )
    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert json.loads(captured.err)["field"] == "model_kind"


@pytest.mark.parametrize("content", [None, "{not json"])
def test_an_unreadable_request_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: str | None
) -> None:
    """A missing request file and one that is not JSON are both refused before any job."""
    path = tmp_path / "request.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    status = run_cli("train", str(path), "--job-root", str(tmp_path / "jobs"))
    captured = capsys.readouterr()
    assert status == 2
    assert captured.out == ""
    assert json.loads(captured.err)["error"] == "training_request_unreadable"
