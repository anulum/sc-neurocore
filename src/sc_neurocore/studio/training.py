# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training service backend for Studio (Block 4)

from __future__ import annotations

import json
import math
import secrets
import threading
import time
import queue
from collections.abc import Callable, Mapping
from io import BytesIO
from typing import Any, cast

from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobCancelled,
    StudioJobContext,
    StudioJobManager,
    StudioJobRejected,
)
from sc_neurocore.studio.platform.action_evidence import (
    EvidenceStatus,
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.training_checkpoint import (
    build_training_checkpoint,
    import_training_checkpoint_payload,
)
from sc_neurocore.studio.platform.training_evidence import build_training_evidence_summary
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_OWNER,
    TRAINING_WEIGHT_ARTIFACT_PATH,
    TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
    build_training_weight_restore_plan,
    training_architecture_fingerprint,
    write_training_weight_checkpoint,
)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_SURROGATES = [
    "fast_sigmoid",
    "superspike",
    "atan_surrogate",
    "sigmoid_surrogate",
    "straight_through",
    "triangular",
]

_CELL_TYPES = [
    "LIFCell",
    "IFCell",
    "ALIFCell",
    "ExpIFCell",
    "AdExCell",
    "LapicqueCell",
    "AlphaCell",
    "SecondOrderLIFCell",
    "RecurrentLIFCell",
]

TRAINING_EVENT_LOG_ARTIFACT_PATH = "training/events.jsonl"
_PERSISTED_TRAINING_EVENT_TYPES = frozenset({"config", "epoch", "completed", "stopped", "error"})


def list_surrogates() -> list[dict[str, Any]]:
    """Return available surrogate-gradient functions for the Studio UI."""

    return [{"name": s, "available": HAS_TORCH} for s in _SURROGATES]


def list_cell_types() -> list[dict[str, Any]]:
    """Return available training cell types for the Studio UI."""

    return [{"name": c, "available": HAS_TORCH} for c in _CELL_TYPES]


class TrainingJob:
    """Manage one Studio training run for thread or process execution."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        job_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        initial_state_dict: Mapping[str, object] | None = None,
    ) -> None:
        self.config = config
        self.id = job_id or f"j{secrets.token_hex(6)}"
        self.status = "pending"
        self.metrics: queue.Queue[Any] = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._cancelled = cancelled
        self._event_sink = event_sink
        self._persisted_event_count = 0
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.final_metrics: dict[str, Any] | None = None
        self.weight_checkpoint: dict[str, JsonValue] | None = None
        self._weight_checkpoint_payload: bytes | None = None
        self._weight_checkpoint_architecture: str | None = None
        self._weight_checkpoint_parameter_count: int | None = None
        self._initial_state_dict = initial_state_dict
        self.live_attach_evidence: dict[str, JsonValue] | None = None

    def start(self) -> None:
        """Start this training job in its legacy background thread."""

        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request cooperative cancellation for this training job."""

        self._stop_event.set()

    def run_blocking(self, context: StudioJobContext) -> dict[str, object]:
        """Run this training job inside a bounded Studio job context.

        Parameters
        ----------
        context:
            Studio job execution context used for cancellation and artifact
            publication.

        Returns
        -------
        dict[str, object]
            Path-free terminal training metadata for the Studio job record.

        Raises
        ------
        StudioJobCancelled
            If the platform job manager requested cancellation.
        Exception
            Re-raises training failures after emitting an SSE error event so the
            platform job record also transitions to failed.
        """

        self.status = "running"
        try:
            self._train(context)
        except Exception as exc:
            self.error = str(exc)
            self._emit("error", {"message": str(exc)})
            self.status = "failed"
            self._write_terminal_artifacts(context, evidence_status="failed")
            raise
        if self.status == "stopped" or context.cancelled:
            self._write_terminal_artifacts(context, evidence_status="cancelled")
            raise StudioJobCancelled("Studio training job was stopped.")
        self._write_terminal_artifacts(context, evidence_status="completed")
        return {
            "training_status": self.status,
            "final_metrics": self.final_metrics,
            "weight_checkpoint": self.weight_checkpoint,
        }

    def _write_terminal_artifacts(
        self,
        context: StudioJobContext,
        *,
        evidence_status: EvidenceStatus,
    ) -> None:
        """Write terminal training status and evidence artifacts."""

        if self._persisted_event_count > 0:
            context.publish_existing_artifact(TRAINING_EVENT_LOG_ARTIFACT_PATH)
        self._publish_weight_checkpoint(context)
        status_payload = self._public_status()
        status_artifact = context.write_artifact(
            "training/status.json",
            json.dumps(status_payload, sort_keys=True),
        )
        write_studio_action_evidence_manifest(
            context,
            action_kind="studio.training.run",
            result=status_payload,
            result_artifact=status_artifact,
            evidence_artifact_path="training/evidence.json",
            evidence_classification="training",
            replay_route="POST /api/training/start",
            status=evidence_status,
            error_message=self.error,
        )

    def _emit(self, event_type: str, data: dict[str, Any]) -> None:
        payload = {"event": event_type, "data": data, "timestamp": time.time()}
        try:
            self.metrics.put_nowait(payload)
        except queue.Full:
            try:
                self.metrics.get_nowait()
            except queue.Empty:
                pass
            self.metrics.put_nowait(payload)
        if self._event_sink is not None and event_type in _PERSISTED_TRAINING_EVENT_TYPES:
            self._event_sink(_json_event_payload(payload))
            self._persisted_event_count += 1

    def _run(self) -> None:
        try:
            self._train()
        except Exception as e:
            self.error = str(e)
            self._emit("error", {"message": str(e)})
            self.status = "failed"

    def _stop_requested(self) -> bool:
        return self._stop_event.is_set() or (self._cancelled is not None and self._cancelled())

    def _public_status(self) -> dict[str, Any]:
        return {
            "error": self.error,
            "final_metrics": self.final_metrics,
            "job_id": self.id,
            "status": self.status,
            "weight_checkpoint": self.weight_checkpoint,
        }

    def _train(self, context: StudioJobContext | None = None) -> None:
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed. pip install sc-neurocore[research]")

        from sc_neurocore.training import (
            SpikingNet,
            SpikeMonitor,
            auto_device,
            model_info,
            spike_count_loss,
        )
        from sc_neurocore.training import surrogate as surr_mod

        cfg = self.config
        dataset = cfg.get("dataset", "synthetic")
        n_epochs = cfg.get("epochs", 10)
        batch_size = cfg.get("batch_size", 64)
        lr = cfg.get("lr", 1e-3)
        hidden = cfg.get("hidden", [128])
        n_timesteps = cfg.get("timesteps", 25)
        surrogate_name = cfg.get("surrogate", "atan_surrogate")
        learn_beta = cfg.get("learn_beta", False)
        learn_threshold = cfg.get("learn_threshold", False)
        max_grad_norm = cfg.get("max_grad_norm", 1.0)

        surrogate_fn = getattr(surr_mod, surrogate_name, surr_mod.atan_surrogate)
        device = auto_device()

        if dataset == "mnist":
            train_loader, test_loader, n_inputs, n_outputs = _load_mnist(batch_size)
        else:
            train_loader, test_loader, n_inputs, n_outputs = _make_synthetic(batch_size)

        n_hidden = hidden[0] if hidden else 128
        n_layers = len(hidden)
        model = SpikingNet(
            n_input=n_inputs,
            n_hidden=n_hidden,
            n_output=n_outputs,
            n_layers=n_layers,
            surrogate_fn=surrogate_fn,
            learn_beta=learn_beta,
            learn_threshold=learn_threshold,
        ).to(device)
        if self._initial_state_dict is not None:
            self._attach_initial_state_dict(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        monitor = SpikeMonitor(model)
        info = model_info(model)

        self._emit(
            "config",
            {
                "job_id": self.id,
                "device": str(device),
                "model_info": info,
                "dataset": dataset,
                "n_epochs": n_epochs,
                "architecture": f"{n_inputs}→{'→'.join(str(n_hidden) for _ in range(n_layers))}→{n_outputs}",
            },
        )

        for epoch in range(n_epochs):
            if self._stop_requested():
                self.status = "stopped"
                self._emit("stopped", {"epoch": epoch})
                return

            if context is not None:
                self._poll_live_attach(context, model, epoch)

            model.train()
            monitor.reset()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                if self._stop_requested():
                    break

                data, targets = data.to(device), targets.to(device)
                data = data.view(data.shape[0], -1)
                data = data.unsqueeze(0).expand(n_timesteps, *data.shape)

                spike_counts, _ = model(data)
                loss = spike_count_loss(spike_counts, targets)

                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call, unused-ignore]
                if max_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                epoch_loss += loss.item() * targets.shape[0]
                correct += (spike_counts.argmax(dim=1) == targets).sum().item()
                total += targets.shape[0]

                if (batch_idx + 1) % 10 == 0:
                    self._emit(
                        "batch",
                        {
                            "epoch": epoch,
                            "batch": batch_idx + 1,
                            "loss": loss.item(),
                            "accuracy": correct / total,
                        },
                    )

            train_loss = epoch_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            # Eval
            model.eval()
            eval_loss = 0.0
            eval_correct = 0
            eval_total = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    data, targets = data.to(device), targets.to(device)
                    data = data.view(data.shape[0], -1)
                    data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
                    spike_counts, _ = model(data)
                    loss = spike_count_loss(spike_counts, targets)
                    eval_loss += loss.item() * targets.shape[0]
                    eval_correct += (spike_counts.argmax(dim=1) == targets).sum().item()
                    eval_total += targets.shape[0]

            val_loss = eval_loss / max(eval_total, 1)
            val_acc = eval_correct / max(eval_total, 1)

            # Layer spike rates from monitor
            layer_rates = {}
            for name in monitor.layer_names:
                raster = monitor.get(name)
                if raster is not None:
                    layer_rates[name] = float(raster.float().mean().item())

            # Parameter snapshots (beta, threshold)
            param_snapshot = {}
            for pname, p in model.named_parameters():
                if "beta_logit" in pname:
                    param_snapshot[pname] = float(torch.sigmoid(p).mean().item())
                elif "threshold_log" in pname:
                    param_snapshot[pname] = float(torch.exp(p).mean().item())

            self._emit(
                "epoch",
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_accuracy": round(train_acc, 4),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 4),
                    "layer_spike_rates": layer_rates,
                    "param_snapshot": param_snapshot,
                },
            )

            monitor.reset()

        self.status = "completed"
        self.final_metrics = {
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
        }
        hidden_architecture = "->".join(str(n_hidden) for _ in range(n_layers))
        architecture = (
            f"{n_inputs}->{hidden_architecture}->{n_outputs}"
            if hidden_architecture
            else f"{n_inputs}->{n_outputs}"
        )
        self._capture_weight_checkpoint(
            model=model,
            architecture=architecture,
            model_info=info,
        )
        self._emit("completed", self.final_metrics)
        monitor.remove()

    def _attach_initial_state_dict(self, model: Any) -> None:
        """Load externally restored weights into the model before training.

        The attach happens at the epoch-zero checkpoint boundary, before any
        optimization step. A strict load fails closed: an architecture mismatch
        raises before training begins, so partial weights are never applied.
        """

        state_dict = self._initial_state_dict
        if state_dict is None:
            return
        try:
            model.load_state_dict(dict(state_dict), strict=True)
        except (RuntimeError, KeyError, ValueError) as exc:
            raise ValueError(
                "Training weight attach is incompatible with the target architecture."
            ) from exc
        self._emit("attach", {"loaded_key_count": len(state_dict)})

    def _poll_live_attach(self, context: StudioJobContext, model: Any, epoch: int) -> None:
        """Consume and apply a pending live weight-attach control command.

        Polled at each epoch boundary. A malformed or incompatible command is
        rejected without interrupting the running job, so a bad attach can never
        crash an in-flight training run.
        """

        try:
            command = context.poll_control_command()
        except ValueError:
            self._emit("attach_rejected", {"epoch": epoch, "reason": "invalid_command"})
            return
        if command is None or command.get("action") != "attach_weights":
            return
        self._apply_live_attach(context, model, command, epoch)

    def _apply_live_attach(
        self,
        context: StudioJobContext,
        model: Any,
        command: Mapping[str, object],
        epoch: int,
    ) -> None:
        """Verify and load a live weight attach, rejecting on any failure."""

        from sc_neurocore.studio.platform.training_weight_loader import (
            load_training_weight_state_dict,
        )
        from sc_neurocore.studio.platform.training_weights import (
            TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
            build_training_weight_restore_attach_evidence,
            materialize_training_weight_payload,
        )

        restore_plan = command.get("restore_plan")
        fingerprint = command.get("architecture_fingerprint")
        weights_seed = command.get("weights_seed_path")
        metadata_seed = command.get("metadata_seed_path")
        if (
            not isinstance(restore_plan, Mapping)
            or not isinstance(fingerprint, str)
            or not isinstance(weights_seed, str)
            or not isinstance(metadata_seed, str)
        ):
            self._emit("attach_rejected", {"epoch": epoch, "reason": "invalid_command"})
            return
        try:
            metadata_payload = context.read_control_seed(metadata_seed)
            weights_payload = context.read_control_seed(weights_seed)
            materialization = materialize_training_weight_payload(
                restore_plan=restore_plan,
                metadata_payload=metadata_payload,
                weights_payload=weights_payload,
                trusted_loader=load_training_weight_state_dict,
            )
            model.load_state_dict(dict(materialization.state_dict), strict=True)
        except (RuntimeError, KeyError, ValueError, StudioJobArtifactUnavailable):
            self._emit("attach_rejected", {"epoch": epoch, "reason": "incompatible"})
            return
        evidence = build_training_weight_restore_attach_evidence(
            materialization,
            mode="live",
            target_job_id=self.id,
            target_architecture=materialization.architecture,
            target_parameter_count=materialization.parameter_count,
            architecture_fingerprint=fingerprint,
        )
        context.write_artifact(
            TRAINING_WEIGHT_RESTORE_ATTACH_EVIDENCE_ARTIFACT_PATH,
            json.dumps(evidence, sort_keys=True),
        )
        self.live_attach_evidence = evidence
        self._emit(
            "attach",
            {
                "epoch": epoch,
                "mode": "live",
                "loaded_key_count": len(materialization.state_dict),
            },
        )

    def _capture_weight_checkpoint(
        self,
        *,
        model: Any,
        architecture: str,
        model_info: dict[str, Any],
    ) -> None:
        """Serialize terminal model weights for later job artifact publication."""

        payload = {
            "config": self.config,
            "final_metrics": self.final_metrics,
            "model_info": model_info,
            "model_state_dict": model.state_dict(),
            "schema_version": STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
        }
        buffer = BytesIO()
        torch.save(payload, buffer)
        self._weight_checkpoint_payload = buffer.getvalue()
        self._weight_checkpoint_architecture = architecture
        self._weight_checkpoint_parameter_count = int(
            sum(parameter.numel() for parameter in model.parameters())
        )

    def _publish_weight_checkpoint(self, context: StudioJobContext) -> None:
        """Publish captured terminal weights into the job artifact manifest."""

        if self._weight_checkpoint_payload is None:
            return
        if self._weight_checkpoint_architecture is None:
            raise ValueError("Training weight checkpoint architecture is missing.")
        if self._weight_checkpoint_parameter_count is None:
            raise ValueError("Training weight checkpoint parameter count is missing.")
        self.weight_checkpoint = write_training_weight_checkpoint(
            context,
            weights_payload=self._weight_checkpoint_payload,
            config=self.config,
            architecture=self._weight_checkpoint_architecture,
            parameter_count=self._weight_checkpoint_parameter_count,
            final_metrics=self.final_metrics,
        ).to_public_dict()


def _make_synthetic(batch_size: int) -> Any:
    """Generate synthetic classification data for quick demos."""
    import torch

    n_samples = 512
    n_inputs = 64
    n_classes = 10
    x = torch.randn(n_samples, n_inputs)
    y = torch.randint(0, n_classes, (n_samples,))
    split = int(0.8 * n_samples)
    train_ds = TensorDataset(x[:split], y[:split])
    test_ds = TensorDataset(x[split:], y[split:])
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_ds, batch_size=batch_size, drop_last=True),
        n_inputs,
        n_classes,
    )


def _load_mnist(batch_size: int) -> Any:
    """Load MNIST via torchvision if available, else synthetic fallback."""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_ds = datasets.MNIST("~/.cache/mnist", train=True, download=True, transform=transform)
        test_ds = datasets.MNIST("~/.cache/mnist", train=False, transform=transform)
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(test_ds, batch_size=batch_size),
            784,
            10,
        )
    except ImportError:
        return _make_synthetic(batch_size)


# Global job registry
_jobs: dict[str, TrainingJob] = {}
_jobs_lock = threading.Lock()


def _register_job(job: TrainingJob) -> None:
    with _jobs_lock:
        _jobs[job.id] = job


def start_training(
    config: dict[str, Any],
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Start a Studio training job.

    When ``job_manager`` is supplied, execution is delegated to the bounded
    Studio job sandbox. Without it, the legacy in-process training thread is
    used for direct module tests and backward compatibility.
    """

    if job_manager is not None:
        from sc_neurocore.studio.platform.training_process import TRAINING_PROCESS_TASK

        record = job_manager.submit_process_task(
            kind="training",
            owner="studio-training",
            request_id=None,
            task_path=TRAINING_PROCESS_TASK,
            payload=config,
        )
        proxy = TrainingJob(config, job_id=record.job_id)
        proxy.status = "running"
        _register_job(proxy)
        return {"job_id": record.job_id, "status": "running"}

    job = TrainingJob(config)
    _register_job(job)
    job.start()
    return {"job_id": job.id, "status": "running"}


def start_training_attach(
    source_job_id: str,
    config: dict[str, Any],
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Start a warm-start training job seeded with restored, verified weights.

    Builds the canonical restore plan from the source training job's stored
    checkpoint metadata, fetches the integrity-checked weight and metadata
    artifacts, and submits a bounded process job that materializes and verifies
    the weights before loading them into the target model at the epoch-zero
    checkpoint boundary. The verified weights are delivered to the worker as
    confined seed inputs, never through the API response.

    Parameters
    ----------
    source_job_id:
        Studio job ID of a completed training job that published weights.
    config:
        Target training configuration for the warm-started job.
    job_manager:
        Bounded Studio job manager that owns artifact reads and job submission.
    expected_config_sha256:
        Optional source configuration digest that the checkpoint must match.

    Returns
    -------
    dict[str, Any]
        ``{"job_id", "status"}`` for the warm-started job, or ``{"error"}`` when
        the source job is unknown or published no usable weight checkpoint.

    Raises
    ------
    ValueError
        If the restore plan cannot be built from the source checkpoint.
    """

    from sc_neurocore.studio.platform.training_process import (
        TRAINING_ATTACH_PROCESS_TASK,
        TRAINING_ATTACH_SEED_METADATA_PATH,
        TRAINING_ATTACH_SEED_WEIGHTS_PATH,
    )

    status_payload = get_training_status(source_job_id, job_manager)
    if "status" not in status_payload:
        return {"error": "training_job_not_found"}
    weight_checkpoint = status_payload.get("weight_checkpoint")
    if not isinstance(weight_checkpoint, dict):
        return {"error": "training_weight_checkpoint_unavailable"}
    source_status = status_payload.get("status")
    if not isinstance(source_status, str) or not source_status:
        return {"error": "training_status_unavailable"}

    restore_plan = build_training_weight_restore_plan(
        source_job_id=source_job_id,
        source_status=source_status,
        weight_checkpoint=weight_checkpoint,
        expected_config_sha256=expected_config_sha256,
    )
    try:
        metadata_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        ).payload
        weights_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_ARTIFACT_PATH,
        ).payload
    except KeyError:
        return {"error": "training_weight_artifact_not_found"}
    except (StudioJobArtifactUnavailable, ValueError):
        return {"error": "training_weight_artifact_unavailable"}

    fingerprint = training_architecture_fingerprint(config)
    record = job_manager.submit_process_task(
        kind="training",
        owner=STUDIO_TRAINING_WEIGHT_RESTORE_ATTACH_OWNER,
        request_id=None,
        task_path=TRAINING_ATTACH_PROCESS_TASK,
        payload={
            "config": config,
            "restore_plan": restore_plan.to_public_dict(),
            "architecture_fingerprint": fingerprint,
        },
        seed_inputs={
            TRAINING_ATTACH_SEED_WEIGHTS_PATH: weights_bytes,
            TRAINING_ATTACH_SEED_METADATA_PATH: metadata_bytes,
        },
    )
    proxy = TrainingJob(config, job_id=record.job_id)
    proxy.status = "running"
    _register_job(proxy)
    return {
        "job_id": record.job_id,
        "status": "running",
        "source_job_id": source_job_id,
        "architecture_fingerprint": fingerprint,
    }


_LIVE_ATTACH_WEIGHTS_SEED = "model_state.pt"
_LIVE_ATTACH_METADATA_SEED = "model_state.json"


def request_live_training_weight_attach(
    target_job_id: str,
    source_job_id: str,
    job_manager: StudioJobManager,
    *,
    expected_config_sha256: str | None = None,
) -> dict[str, Any]:
    """Deliver verified weights to a running training job for a live attach.

    Validates that the target job is running, that the source job published a
    weight checkpoint, and that the source and target architectures are
    compatible, then delivers the integrity-checked weight artifacts to the
    running worker as a confined control command. The worker applies the attach
    at its next epoch boundary; an incompatible attach is rejected without
    interrupting the running job.

    Parameters
    ----------
    target_job_id:
        Studio job ID of the running training job to attach weights into.
    source_job_id:
        Studio job ID of a completed training job that published weights.
    job_manager:
        Bounded Studio job manager that owns artifact reads and control delivery.
    expected_config_sha256:
        Optional source configuration digest that the checkpoint must match.

    Returns
    -------
    dict[str, Any]
        ``{"target_job_id", "status", "architecture_fingerprint"}`` when the
        attach command is delivered, or ``{"error"}`` for a precondition failure.

    Raises
    ------
    ValueError
        If the restore plan cannot be built from the source checkpoint.
    """

    try:
        target_record = job_manager.record(target_job_id)
    except KeyError:
        return {"error": "training_job_not_found"}
    if target_record.status != "running":
        return {"error": "training_job_not_running"}
    with _jobs_lock:
        target_proxy = _jobs.get(target_job_id)
        source_proxy = _jobs.get(source_job_id)
    target_config = dict(target_proxy.config) if target_proxy is not None else {}

    source_status = get_training_status(source_job_id, job_manager)
    if "status" not in source_status:
        return {"error": "source_job_not_found"}
    weight_checkpoint = source_status.get("weight_checkpoint")
    if not isinstance(weight_checkpoint, dict):
        return {"error": "training_weight_checkpoint_unavailable"}
    source_job_status = source_status.get("status")
    if not isinstance(source_job_status, str) or not source_job_status:
        return {"error": "training_status_unavailable"}

    restore_plan = build_training_weight_restore_plan(
        source_job_id=source_job_id,
        source_status=source_job_status,
        weight_checkpoint=weight_checkpoint,
        expected_config_sha256=expected_config_sha256,
    )
    try:
        metadata_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_METADATA_ARTIFACT_PATH,
        ).payload
        weights_bytes = job_manager.read_artifact(
            source_job_id,
            TRAINING_WEIGHT_ARTIFACT_PATH,
        ).payload
    except KeyError:
        return {"error": "training_weight_artifact_not_found"}
    except (StudioJobArtifactUnavailable, ValueError):
        return {"error": "training_weight_artifact_unavailable"}

    fingerprint = training_architecture_fingerprint(target_config)
    if source_proxy is not None and (
        training_architecture_fingerprint(source_proxy.config) != fingerprint
    ):
        return {"error": "architecture_incompatible"}

    try:
        job_manager.send_control_command(
            target_job_id,
            command={
                "action": "attach_weights",
                "restore_plan": restore_plan.to_public_dict(),
                "architecture_fingerprint": fingerprint,
                "weights_seed_path": _LIVE_ATTACH_WEIGHTS_SEED,
                "metadata_seed_path": _LIVE_ATTACH_METADATA_SEED,
            },
            seed_inputs={
                _LIVE_ATTACH_WEIGHTS_SEED: weights_bytes,
                _LIVE_ATTACH_METADATA_SEED: metadata_bytes,
            },
        )
    except StudioJobRejected:
        return {"error": "training_job_not_running"}
    return {
        "target_job_id": target_job_id,
        "source_job_id": source_job_id,
        "status": "attach_requested",
        "architecture_fingerprint": fingerprint,
    }


def stop_training(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Request cooperative stop for a Studio training job."""

    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    job.stop()
    if job_manager is not None:
        try:
            job_manager.cancel(job_id)
        except KeyError:
            pass
    return {"job_id": job_id, "status": "stopping"}


def get_training_status(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Return path-free status for one Studio training job."""

    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                pass
            else:
                return _status_from_platform_record(
                    record,
                    evidence_summary=build_training_evidence_summary(
                        record,
                        job_manager.read_artifact,
                    ),
                )
        return {"error": f"Job {job_id} not found"}
    if job_manager is not None:
        try:
            record = job_manager.record(job_id)
        except KeyError:
            return job._public_status()
        _sync_proxy_job(job, record.status, record.error, record.result)
        return _status_with_evidence_summary(
            job._public_status(),
            build_training_evidence_summary(record, job_manager.read_artifact),
        )
    return job._public_status()


def stream_metrics(job_id: str, job_manager: StudioJobManager | None = None) -> Any:
    """Generator that yields SSE-formatted metric events."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                record = None
            if record is not None:
                yield f"data: {json.dumps(_event_from_platform_record(record.status, record.error, record.result))}\n\n"
                return
        yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Job not found'}})}\n\n"
        return

    live_event_offset = 0
    live_event_buffer = ""
    live_terminal_seen = False
    while True:
        if job_manager is not None:
            try:
                record = job_manager.record(job_id)
            except KeyError:
                record = None
            if record is not None:
                _sync_proxy_job(job, record.status, record.error, record.result)
                live_events, live_event_offset, live_event_buffer = _read_live_training_events(
                    job_manager,
                    job_id,
                    offset=live_event_offset,
                    buffer=live_event_buffer,
                )
                for event in live_events:
                    if event.get("event") in ("completed", "stopped", "error"):
                        live_terminal_seen = True
                    yield f"data: {json.dumps(event)}\n\n"
                if job.status in ("completed", "stopped", "failed"):
                    if not live_terminal_seen:
                        yield f"data: {json.dumps(_event_from_platform_record(record.status, record.error, record.result))}\n\n"
                    break
        try:
            event = job.metrics.get(timeout=1.0)
            yield f"data: {json.dumps(event)}\n\n"
            if event["event"] in ("completed", "stopped", "error"):
                break
        except queue.Empty:
            if job.status in ("completed", "stopped", "failed"):
                break
            yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"


def list_jobs() -> list[dict[str, Any]]:
    """Return path-free summaries for known Studio training jobs."""

    with _jobs_lock:
        return [{"job_id": j.id, "status": j.status, "config": j.config} for j in _jobs.values()]


def export_training_checkpoint(
    job_id: str,
    job_manager: StudioJobManager | None = None,
) -> dict[str, Any]:
    """Return a portable checkpoint for one Studio training job.

    Parameters
    ----------
    job_id:
        Training Monitor job identifier.
    job_manager:
        Optional Studio job manager used to attach terminal worker evidence
        metadata when the job has reached a terminal state.

    Returns
    -------
    dict[str, Any]
        `studio.training.checkpoint.v1` payload, or an error payload when the
        training job is unknown to the parent-process Training Monitor
        registry.
    """

    with _jobs_lock:
        job = _jobs.get(job_id)
    if job is None:
        return {"error": f"Job {job_id} not found"}
    status = get_training_status(job_id, job_manager)
    final_metrics = status.get("final_metrics")
    evidence_summary = status.get("evidence_summary")
    weight_checkpoint = status.get("weight_checkpoint")
    checkpoint = build_training_checkpoint(
        job_id=job_id,
        config=job.config,
        status=str(status.get("status", job.status)),
        final_metrics=final_metrics if isinstance(final_metrics, dict) else None,
        evidence_summary=evidence_summary if isinstance(evidence_summary, dict) else None,
        weight_checkpoint=weight_checkpoint if isinstance(weight_checkpoint, dict) else None,
    )
    return checkpoint.to_public_dict()


def import_training_checkpoint(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a portable checkpoint and return its training config.

    Parameters
    ----------
    data:
        JSON object submitted to `/api/training/checkpoint/import`.

    Returns
    -------
    dict[str, Any]
        Validated checkpoint import payload containing restored training
        configuration and source-job metadata.

    Raises
    ------
    ValueError
        If the checkpoint schema, config digest, or checkpoint digest is
        invalid.
    """

    return import_training_checkpoint_payload(data)


def _sync_proxy_job(
    job: TrainingJob,
    platform_status: str,
    platform_error: str | None,
    platform_result: dict[str, object] | None,
) -> None:
    """Update a parent-process proxy job from platform job terminal state."""

    if platform_status == "completed":
        job.status = "completed"
        final_metrics = (platform_result or {}).get("final_metrics")
        if isinstance(final_metrics, dict):
            job.final_metrics = dict(final_metrics)
        weight_checkpoint = (platform_result or {}).get("weight_checkpoint")
        if isinstance(weight_checkpoint, dict):
            job.weight_checkpoint = cast(dict[str, JsonValue], dict(weight_checkpoint))
        return
    if platform_status in ("cancelled", "cancelling", "timed_out"):
        job.status = "stopped"
        job.error = platform_error
        return
    if platform_status == "failed":
        job.status = "failed"
        job.error = platform_error


def _status_from_platform_record(
    record: Any,
    *,
    evidence_summary: dict[str, object] | None = None,
) -> dict[str, Any]:
    """Return Training Monitor status synthesized from a platform job record."""

    platform_result = record.result if isinstance(record.result, dict) else None
    final_metrics = (platform_result or {}).get("final_metrics")
    weight_checkpoint = (platform_result or {}).get("weight_checkpoint")
    return _status_with_evidence_summary(
        {
            "error": record.error,
            "final_metrics": final_metrics if isinstance(final_metrics, dict) else None,
            "job_id": record.job_id,
            "status": _training_status_from_platform_status(record.status),
            "weight_checkpoint": weight_checkpoint if isinstance(weight_checkpoint, dict) else None,
        },
        evidence_summary,
    )


def _status_with_evidence_summary(
    status: dict[str, Any],
    evidence_summary: dict[str, object] | None,
) -> dict[str, Any]:
    """Attach path-free evidence metadata to a public training status."""

    if evidence_summary is None:
        return status
    return {
        **status,
        "evidence_summary": evidence_summary,
    }


def _event_from_platform_record(
    platform_status: str,
    platform_error: str | None,
    platform_result: dict[str, object] | None,
) -> dict[str, Any]:
    """Return an SSE event synthesized from a platform job record."""

    training_status = _training_status_from_platform_status(platform_status)
    if training_status == "completed":
        final_metrics = (platform_result or {}).get("final_metrics")
        return {
            "data": final_metrics if isinstance(final_metrics, dict) else {},
            "event": "completed",
            "timestamp": time.time(),
        }
    if training_status == "failed":
        return {
            "data": {"message": platform_error or "Training failed."},
            "event": "error",
            "timestamp": time.time(),
        }
    if training_status == "stopped":
        return {
            "data": {"message": platform_error or "Training stopped."},
            "event": "stopped",
            "timestamp": time.time(),
        }
    return {"event": "heartbeat"}


def _training_status_from_platform_status(platform_status: str) -> str:
    """Map platform job status into the Training Monitor status vocabulary."""

    if platform_status == "completed":
        return "completed"
    if platform_status == "failed":
        return "failed"
    if platform_status in ("cancelled", "cancelling", "timed_out"):
        return "stopped"
    if platform_status in ("pending", "running"):
        return "running"
    return "pending"


def _json_event_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return a JSON-compatible copy of a Training Monitor SSE event."""

    converted = _json_compatible(payload)
    if not isinstance(converted, dict):
        raise ValueError("Training event payload must remain a JSON object.")
    return cast(dict[str, object], converted)


def _json_compatible(value: object) -> Any:
    """Return ``value`` converted to JSON-compatible containers."""

    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_compatible(item) for item in value]
    return str(value)


def _read_live_training_events(
    job_manager: StudioJobManager,
    job_id: str,
    *,
    offset: int,
    buffer: str,
) -> tuple[list[dict[str, object]], int, str]:
    """Read complete JSONL Training Monitor events appended by a process worker."""

    payload, new_offset = job_manager.read_live_artifact_bytes(
        job_id,
        TRAINING_EVENT_LOG_ARTIFACT_PATH,
        offset=offset,
    )
    if not payload:
        return [], new_offset, buffer
    text = buffer + payload.decode("utf-8")
    lines = text.splitlines(keepends=True)
    next_buffer = ""
    if lines and not lines[-1].endswith("\n"):
        next_buffer = lines.pop()
    events: list[dict[str, object]] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(dict(event))
    return events, new_offset, next_buffer
