# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Command-line Studio training acceptance

"""Run ``sc-neurocore train`` as a user would and check what it reports and returns.

Every case starts the real console module in its own process, which submits
the request through the Studio contract and job manager, runs the worker and
seals the job under the given job root.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
_FAST_RUN = {"dataset": "synthetic", "epochs": 1, "batch_size": 32, "timesteps": 4}


def train(tmp_path: Path, request: object, *extra: str) -> subprocess.CompletedProcess[str]:
    """Write a request and run the command on it.

    Parameters
    ----------
    tmp_path:
        Directory for the request file and the job root.
    request:
        The request body, or a string written verbatim.
    extra:
        Further command arguments.

    Returns
    -------
    subprocess.CompletedProcess
        The finished command.
    """
    path = tmp_path / "request.json"
    path.write_text(request if isinstance(request, str) else json.dumps(request))
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "sc_neurocore.cli",
            "train",
            str(path),
            "--job-root",
            str(tmp_path / "jobs"),
            *extra,
        ],
        cwd=tmp_path,
        env=dict(
            os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")
        ),
        capture_output=True,
        text=True,
        timeout=600,
    )


def test_a_met_criterion_completes_with_status_zero(tmp_path: Path) -> None:
    """The verdict, metrics and sealed checkpoint are printed; the job is kept on disk."""
    request = {**_FAST_RUN, "preregistration": {"metric": "val_accuracy", "threshold": 0.0}}
    result = train(tmp_path, request)
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "completed" and outcome["error"] is None
    assert outcome["preregistration_verdict"]["passed"] is True
    assert set(outcome["final_metrics"]) == {
        "train_loss",
        "train_accuracy",
        "val_loss",
        "val_accuracy",
    }
    assert outcome["weight_checkpoint"]["config_sha256"]
    assert outcome["job_root"] == str((tmp_path / "jobs").resolve())
    assert any((tmp_path / "jobs").iterdir())


def test_a_missed_criterion_completes_with_its_own_status(tmp_path: Path) -> None:
    """A run that misses its declared criterion is still a completed run, reported as missed."""
    result = train(
        tmp_path, {**_FAST_RUN, "preregistration": {"metric": "val_loss", "threshold": 0.0}}
    )
    assert result.returncode == 3, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "completed"
    assert outcome["preregistration_verdict"]["passed"] is False


def test_a_run_without_a_criterion_completes_with_status_zero(tmp_path: Path) -> None:
    """Declaring no criterion is allowed and judged by nothing."""
    result = train(tmp_path, _FAST_RUN)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)["preregistration_verdict"] is None


def test_a_conversion_run_reports_the_converted_network(tmp_path: Path) -> None:
    """The conversion route's metrics and its accuracy-drop verdict reach the command's output."""
    request = {
        **_FAST_RUN,
        "model_kind": "qcfs_conversion",
        "hidden": [16],
        "target_profile": "loihi2",
        "preregistration": {"metric": "conversion_accuracy_drop", "threshold": 1.0},
    }
    result = train(tmp_path, request)
    assert result.returncode == 0, result.stdout + result.stderr
    outcome = json.loads(result.stdout)
    assert outcome["status"] == "completed"
    metrics = outcome["final_metrics"]
    assert {
        "source_val_accuracy",
        "conversion_accuracy_drop",
        "val_accuracy",
        "target_accuracy",
    } <= set(metrics)
    verdict = outcome["preregistration_verdict"]
    assert verdict["metric"] == "conversion_accuracy_drop" and verdict["passed"] is True


def test_a_conversion_request_on_event_data_is_refused(tmp_path: Path) -> None:
    """The conversion route names the static datasets it can encode."""
    result = train(tmp_path, {**_FAST_RUN, "model_kind": "qcfs_conversion", "dataset": "shd"})
    assert result.returncode == 2
    detail = json.loads(result.stderr)
    assert detail["field"] == "dataset" and detail["supported"] == ["synthetic", "mnist"]


def test_a_refused_request_names_its_field_and_starts_nothing(tmp_path: Path) -> None:
    """The contract refusal is printed on stderr and no job is created."""
    result = train(
        tmp_path, {**_FAST_RUN, "preregistration": {"metric": "val_accuracy", "threshold": 7}}
    )
    assert result.returncode == 2
    detail = json.loads(result.stderr)
    assert detail["error"] == "training_config_rejected" and detail["field"] == "preregistration"
    assert result.stdout == ""


def test_an_unreadable_request_is_refused(tmp_path: Path) -> None:
    """A request file that is not JSON is refused before a job manager exists."""
    result = train(tmp_path, "{not json")
    assert result.returncode == 2
    assert json.loads(result.stderr)["error"] == "training_request_unreadable"
    assert not (tmp_path / "jobs").exists()


def test_a_run_that_outlives_its_timeout_reports_failure(tmp_path: Path) -> None:
    """A job the manager times out is not reported as a completed run."""
    request = {
        **_FAST_RUN,
        "epochs": 200,
        "preregistration": {"metric": "val_accuracy", "threshold": 0.0},
    }
    result = train(tmp_path, request, "--timeout", "0.5")
    assert result.returncode == 1, result.stdout + result.stderr
    assert json.loads(result.stdout)["status"] != "completed"
