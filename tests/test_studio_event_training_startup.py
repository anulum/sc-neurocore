# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Fresh event training and checkpoint runtime ordering

"""Exercise actual public training tasks in fresh, resource-limited processes."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.datasets.encoders import EventBinning
from sc_neurocore.datasets.manifest import build_manifest
from sc_neurocore.datasets.splits import group_split
from sc_neurocore.studio.event_training_contract import EventTrainingContract
from sc_neurocore.studio.platform.jobs import STUDIO_SEED_INPUT_DIR
from sc_neurocore.studio.platform.training_weights import (
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
)
from tests.event_dataset_support import write_nmnist

_CHILD = """
import resource
for kind, limit in (
    (resource.RLIMIT_CORE, 0), (resource.RLIMIT_AS, 4 << 30),
    (resource.RLIMIT_CPU, 600), (resource.RLIMIT_NOFILE, 256),
    (resource.RLIMIT_FSIZE, 64 << 20),
):
    resource.setrlimit(kind, (limit, limit))
import json, sys, threading
from pathlib import Path
from sc_neurocore.studio.platform.jobs import StudioJobContext
from sc_neurocore.studio.platform.training_process import (
    run_training_process_task, run_training_attach_process_task,
)
assert 'torch' not in sys.modules and 'juliacall' not in sys.modules
work = Path(sys.argv[1])
payload = json.loads(Path(sys.argv[2]).read_text())
context = StudioJobContext(job_id='sj_startup', work_dir=work,
    cancel_event=threading.Event(), max_artifact_bytes=1 << 20)
task = run_training_attach_process_task if 'restore_plan' in payload else run_training_process_task
result = task(context, payload)
print(json.dumps({'result': result, 'julia_loaded': 'juliacall' in sys.modules}), flush=True)
"""


def _config(root: Path) -> dict[str, Any]:
    """Bind a small actual camera recording corpus to its complete split."""
    write_nmnist(root, {"train": {0: 3, 1: 2}})
    manifest = build_manifest("nmnist", root, version="generated-format-fixture")
    contract = EventTrainingContract(
        manifest,
        group_split(manifest, fractions={"train": 0.6, "evaluation": 0.4}, seed=7),
        EventBinning(1.0, 4, 34, 34, "merge"),
        "train",
        "evaluation",
    )
    return {
        "dataset": "nmnist",
        "epochs": 1,
        "batch_size": 3,
        "hidden": [4],
        "timesteps": 4,
        "seed": 7,
        "event_data": contract.to_dict(),
    }


def _run(work: Path, payload: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
    """Execute the actual task with the upstream import-order warning made fatal."""
    work.mkdir(exist_ok=True)
    request = work / "request.json"
    request.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-W",
            "error:torch was imported before juliacall:UserWarning",
            "-c",
            _CHILD,
            str(work),
            str(request),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result: dict[str, Any] = json.loads(completed.stdout.splitlines()[-1])
    assert result["result"]["training_status"] == "completed"
    assert result["julia_loaded"] == (env["SC_NEUROCORE_DATASET_JULIA_ENABLED"] == "1")
    training_result = result["result"]
    assert isinstance(training_result, dict)
    return training_result


@pytest.mark.parametrize("backend", ["numpy", "julia"])
def test_fresh_training_warm_start_and_exact_resume_preserve_complete_state(
    tmp_path: Path,
    backend: str,
) -> None:
    """Julia starts before Torch even when trusted restored weights load first."""
    config = _config(tmp_path / "recordings")
    env = dict(os.environ)
    for name in ("RUST", "MOJO", "GO"):
        env.pop(f"SC_NEUROCORE_DATASET_{name}_LIBRARY", None)
    env["SC_NEUROCORE_DATASET_JULIA_ENABLED"] = "1" if backend == "julia" else "0"
    env["SC_NEUROCORE_STUDIO_DATASET_ROOT"] = str(tmp_path / "recordings")
    first = _run(tmp_path / "first", config, env)
    _run(tmp_path / "whole", {**config, "epochs": 2}, env)
    plan = build_training_weight_restore_plan(
        source_job_id="sj_startup",
        source_status="completed",
        weight_checkpoint=first["weight_checkpoint"],
    ).to_public_dict()
    for mode in ("warm_start", "exact_resume"):
        work = tmp_path / mode
        seed = work / STUDIO_SEED_INPUT_DIR
        seed.mkdir(parents=True)
        for filename in ("model_state.pt", "model_state.json"):
            (seed / filename).write_bytes((tmp_path / "first" / "training" / filename).read_bytes())
        target = {**config, "epochs": 2 if mode == "exact_resume" else 1}
        result = _run(
            work,
            {
                "config": target,
                "restore_plan": plan,
                "mode": mode,
                "architecture_fingerprint": training_architecture_fingerprint(target),
            },
            env,
        )
        assert result["weight_restore_attach"]["mode"] == mode
    comparator = Path(__file__).resolve().parents[1] / "studio/frontend/e2e/event_training_state.py"
    subprocess.run(
        [
            sys.executable,
            str(comparator),
            str(tmp_path / "exact_resume/training/model_state.pt"),
            str(tmp_path / "whole/training/model_state.pt"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def test_admission_refuses_changed_manifest_without_importing_compute_runtimes(
    tmp_path: Path,
) -> None:
    """The public service validates custody before Torch, Julia or job execution."""
    config = _config(tmp_path / "recordings")
    next((tmp_path / "recordings").rglob("*.bin")).write_bytes(b"changed")
    request = tmp_path / "request.json"
    request.write_text(json.dumps(config), encoding="utf-8")
    child = """
import json, sys
from pathlib import Path
from sc_neurocore.studio.training import start_training
try:
    start_training(json.loads(Path(sys.argv[1]).read_text()))
except ValueError as error:
    assert str(error) == 'event dataset files or sample metadata differ from the manifest'
else:
    raise AssertionError('changed data was admitted')
assert 'torch' not in sys.modules and 'juliacall' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", child, str(request)],
        env={**os.environ, "SC_NEUROCORE_STUDIO_DATASET_ROOT": str(tmp_path / "recordings")},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )


def test_admission_rechecks_confinement_after_manifest_reads(tmp_path: Path) -> None:
    """A directory redirected during digest reads cannot pass file confinement."""
    root = tmp_path / "recordings"
    config = _config(root)
    directory = root / "Train/0"
    inside = root / "inside"
    directory.rename(inside)
    outside = tmp_path / "outside"
    shutil.copytree(inside, outside)
    directory.symlink_to(inside, target_is_directory=True)
    request = tmp_path / "request.json"
    request.write_text(json.dumps(config), encoding="utf-8")
    child = """
import json, sys
from pathlib import Path
from sc_neurocore.studio.training import start_training
directory, outside = Path(sys.argv[2]), Path(sys.argv[3])
redirected = False
def redirect(event, args):
    global redirected
    if event == 'open' and not redirected and str(args[0]).startswith(str(directory) + '/'):
        redirected = True
        directory.unlink()
        directory.symlink_to(outside, target_is_directory=True)
sys.addaudithook(redirect)
try:
    start_training(json.loads(Path(sys.argv[1]).read_text()))
except ValueError as error:
    assert str(error) == 'event dataset file lies outside the configured root'
else:
    raise AssertionError('redirected data was admitted')
assert redirected
assert 'torch' not in sys.modules and 'juliacall' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", child, str(request), str(directory), str(outside)],
        env={**os.environ, "SC_NEUROCORE_STUDIO_DATASET_ROOT": str(root)},
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
