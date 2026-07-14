# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training-job execution tests

"""Exercise real TrainingJob execution, artifacts, and optional backends."""

from __future__ import annotations

import builtins
import hashlib
import io
import json
import queue
import runpy
import threading
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from sc_neurocore.studio.platform.jobs import StudioJobCancelled, StudioJobContext
from sc_neurocore.studio.platform.training_process import run_training_process_task
from sc_neurocore.studio.platform.training_weights import (
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
)
from sc_neurocore.studio.training import TRAINING_EVENT_LOG_ARTIFACT_PATH, TrainingJob

_SOURCE_PATH = Path(__file__).resolve().parents[1] / "src/sc_neurocore/studio/_training_job.py"
_SYNTHETIC_CONFIG = {
    "dataset": "synthetic",
    "epochs": 1,
    "batch_size": 64,
    "hidden": [8],
    "timesteps": 2,
}


def _context(tmp_path: Path, job_id: str, *, cancelled: bool = False) -> StudioJobContext:
    work_dir = tmp_path / job_id
    work_dir.mkdir()
    cancel_event = threading.Event()
    if cancelled:
        cancel_event.set()
    return StudioJobContext(
        job_id=job_id,
        work_dir=work_dir,
        cancel_event=cancel_event,
        max_artifact_bytes=50_000_000,
    )


def _state_digest(weights_payload: bytes) -> str:
    torch = pytest.importorskip("torch")
    checkpoint = torch.load(io.BytesIO(weights_payload), map_location="cpu", weights_only=True)
    state_dict = checkpoint["model_state_dict"]
    rows = []
    for name in sorted(state_dict):
        tensor = state_dict[name].detach().cpu().contiguous()
        rows.append(
            {
                "dtype": str(tensor.dtype),
                "key": name,
                "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
                "shape": list(tensor.shape),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _wait_for_terminal(job: TrainingJob) -> None:
    deadline = time.monotonic() + 30.0
    while job.status not in {"completed", "failed", "stopped"}:
        if time.monotonic() >= deadline:
            pytest.fail("training job did not reach a terminal state")
        time.sleep(0.01)


def _blocked_import(
    original_import: Callable[..., ModuleType],
    blocked_prefix: str,
) -> Callable[..., ModuleType]:
    def import_module(name: str, *args: object, **kwargs: object) -> ModuleType:
        if name == blocked_prefix or name.startswith(f"{blocked_prefix}."):
            raise ImportError(f"blocked optional dependency: {blocked_prefix}")
        return original_import(name, *args, **kwargs)

    return import_module


def test_seeded_synthetic_training_preserves_metrics_and_weight_artifact(
    tmp_path: Path,
) -> None:
    """A seeded real run pins the parent-compatible metrics and tensor state."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(20260714)
    context = _context(tmp_path, "sj_training_seeded")

    result = run_training_process_task(context, dict(_SYNTHETIC_CONFIG))

    expected_metrics = {
        "train_accuracy": 0.1172,
        "train_loss": 2.302585,
        "val_accuracy": 0.125,
        "val_loss": 2.302585,
    }
    assert result["training_status"] == "completed"
    assert result["final_metrics"] == expected_metrics
    metadata = json.loads(
        (tmp_path / context.job_id / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_text()
    )
    assert metadata["architecture"] == "64->8->10"
    assert metadata["parameter_count"] == 610
    weights = (tmp_path / context.job_id / TRAINING_WEIGHT_ARTIFACT_PATH).read_bytes()
    assert (
        _state_digest(weights) == "988661edac7c35847307bddc2616d3d50ed129deddae06c90c94a7bd37042b8e"
    )
    events = [
        json.loads(line)
        for line in (tmp_path / context.job_id / TRAINING_EVENT_LOG_ARTIFACT_PATH)
        .read_text()
        .splitlines()
    ]
    assert [event["event"] for event in events] == ["config", "epoch", "completed"]


def test_bounded_metric_queue_keeps_latest_event_with_empty_dataloaders() -> None:
    """A saturated metric queue evicts old events without blocking training."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(7)
    job = TrainingJob(
        {
            "dataset": "synthetic",
            "epochs": 1,
            "batch_size": 1024,
            "hidden": [8],
            "timesteps": 1,
            "learn_beta": True,
            "learn_threshold": True,
            "max_grad_norm": 0.0,
        },
        job_id="sj_training_bounded_queue",
    )
    job.metrics = queue.Queue(maxsize=1)

    job.start()
    _wait_for_terminal(job)

    assert job.status == "completed"
    assert job.final_metrics == {
        "train_loss": 0.0,
        "train_accuracy": 0.0,
        "val_loss": 0.0,
        "val_accuracy": 0.0,
    }
    deadline = time.monotonic() + 5.0
    terminal_event: dict[str, object] | None = None
    while terminal_event is None:
        try:
            event = job.metrics.get_nowait()
        except queue.Empty:
            if time.monotonic() >= deadline:
                pytest.fail("training job did not publish its terminal metric event")
            time.sleep(0.01)
            continue
        if event["event"] == "completed":
            terminal_event = cast(dict[str, object], event)
    assert terminal_event["event"] == "completed"


def test_training_job_fails_cleanly_when_torch_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The optional-backend import path reports an actionable terminal error."""
    original_import = cast(Callable[..., ModuleType], builtins.__import__)
    with monkeypatch.context() as patch:
        patch.setattr(builtins, "__import__", _blocked_import(original_import, "torch"))
        namespace = runpy.run_path(
            str(_SOURCE_PATH),
            run_name="sc_neurocore.studio._training_without_torch",
        )
    unavailable_job_type = cast(type[TrainingJob], namespace["TrainingJob"])
    job = unavailable_job_type({"dataset": "synthetic"}, job_id="sj_no_torch")

    job.start()
    _wait_for_terminal(job)

    assert namespace["HAS_TORCH"] is False
    assert job.status == "failed"
    assert job.error == "PyTorch not installed. pip install sc-neurocore[research]"


def test_mnist_adapter_trains_through_a_protocol_compatible_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The MNIST route consumes the torchvision Dataset protocol end to end."""
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")

    def local_mnist(
        _root: str,
        *,
        train: bool,
        download: bool = False,
        transform: object = None,
    ) -> object:
        del download, transform
        sample_count = 8 if train else 4
        features = torch.linspace(0.0, 1.0, sample_count * 28 * 28).reshape(sample_count, 1, 28, 28)
        targets = torch.arange(sample_count) % 10
        return torch.utils.data.TensorDataset(features, targets)

    monkeypatch.setattr(torchvision.datasets, "MNIST", local_mnist)
    torch.manual_seed(11)
    context = _context(tmp_path, "sj_training_mnist")
    job = TrainingJob(
        {
            "dataset": "mnist",
            "epochs": 1,
            "batch_size": 4,
            "hidden": [4],
            "timesteps": 1,
            "max_grad_norm": 0.0,
        },
        job_id=context.job_id,
    )

    result = job.run_blocking(context)

    assert result["training_status"] == "completed"
    metadata = json.loads(
        (tmp_path / context.job_id / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_text()
    )
    assert metadata["architecture"] == "784->4->10"


def test_mnist_adapter_falls_back_when_torchvision_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A missing torchvision install falls back to the documented synthetic data."""
    pytest.importorskip("torch")
    original_import = cast(Callable[..., ModuleType], builtins.__import__)
    context = _context(tmp_path, "sj_training_mnist_fallback")
    job = TrainingJob(
        {
            "dataset": "mnist",
            "epochs": 1,
            "batch_size": 1024,
            "hidden": [8],
            "timesteps": 1,
        },
        job_id=context.job_id,
    )

    with monkeypatch.context() as patch:
        patch.setattr(
            builtins,
            "__import__",
            _blocked_import(original_import, "torchvision"),
        )
        result = job.run_blocking(context)

    assert result["training_status"] == "completed"
    metadata = json.loads(
        (tmp_path / context.job_id / TRAINING_WEIGHT_METADATA_ARTIFACT_PATH).read_text()
    )
    assert metadata["architecture"] == "64->8->10"


def test_initial_state_dict_loads_strictly_and_rejects_incompatible_state(
    tmp_path: Path,
) -> None:
    """Initial weights load through the real model and fail closed on mismatch."""
    torch = pytest.importorskip("torch")
    from sc_neurocore.training import SpikingNet

    model = SpikingNet(n_input=64, n_hidden=8, n_output=10, n_layers=1)
    valid_context = _context(tmp_path, "sj_training_initial_valid")
    valid_job = TrainingJob(
        {**_SYNTHETIC_CONFIG, "batch_size": 1024},
        job_id=valid_context.job_id,
        initial_state_dict=model.state_dict(),
    )

    valid_result = valid_job.run_blocking(valid_context)

    assert valid_result["training_status"] == "completed"
    assert any(event["event"] == "attach" for event in list(valid_job.metrics.queue))

    invalid_context = _context(tmp_path, "sj_training_initial_invalid")
    invalid_job = TrainingJob(
        {**_SYNTHETIC_CONFIG, "batch_size": 1024},
        job_id=invalid_context.job_id,
        initial_state_dict={"not-a-model-key": torch.zeros(1)},
    )
    with pytest.raises(ValueError, match="incompatible with the target architecture"):
        invalid_job.run_blocking(invalid_context)
    evidence = json.loads(
        (tmp_path / invalid_context.job_id / "training/evidence.json").read_text()
    )
    assert evidence["status"] == "failed"


def test_cancellation_is_honoured_at_epoch_and_batch_boundaries(tmp_path: Path) -> None:
    """Process and injected cancellation probes stop at their documented boundaries."""
    pytest.importorskip("torch")
    cancelled_context = _context(tmp_path, "sj_training_cancelled", cancelled=True)
    with pytest.raises(StudioJobCancelled, match="stopped"):
        run_training_process_task(cancelled_context, dict(_SYNTHETIC_CONFIG))
    cancelled_events = [
        json.loads(line)
        for line in (tmp_path / cancelled_context.job_id / TRAINING_EVENT_LOG_ARTIFACT_PATH)
        .read_text()
        .splitlines()
    ]
    assert [event["event"] for event in cancelled_events] == ["config", "stopped"]

    checks = 0

    def cancel_during_first_batch() -> bool:
        nonlocal checks
        checks += 1
        return checks >= 2

    batch_context = _context(tmp_path, "sj_training_batch_cancel")
    batch_job = TrainingJob(
        {**_SYNTHETIC_CONFIG, "timesteps": 1},
        job_id=batch_context.job_id,
        cancelled=cancel_during_first_batch,
    )
    result = batch_job.run_blocking(batch_context)

    assert result["training_status"] == "completed"
    assert cast(dict[str, float], result["final_metrics"])["train_loss"] == 0.0
