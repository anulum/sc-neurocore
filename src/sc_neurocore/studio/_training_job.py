# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training job execution

"""Execute one Studio SNN training run and publish its bounded artifacts."""

from __future__ import annotations

import json
import queue
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Protocol, cast

from sc_neurocore.studio._training_events import (
    TRAINING_EVENT_LOG_ARTIFACT_PATH,
    _json_event_payload,
)
from sc_neurocore.studio.platform.action_evidence import (
    EvidenceStatus,
    write_studio_action_evidence_manifest,
)
from sc_neurocore.studio.platform.evidence_bundle import JsonValue
from sc_neurocore.studio.platform.jobs import (
    StudioJobArtifactUnavailable,
    StudioJobCancelled,
    StudioJobContext,
)
from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    write_training_weight_checkpoint,
)

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


_SURROGATES = (
    "fast_sigmoid",
    "superspike",
    "atan_surrogate",
    "sigmoid_surrogate",
    "straight_through",
    "triangular",
)

_CELL_TYPES = (
    "LIFCell",
    "IFCell",
    "ALIFCell",
    "ExpIFCell",
    "AdExCell",
    "LapicqueCell",
    "AlphaCell",
    "SecondOrderLIFCell",
    "RecurrentLIFCell",
)

_PERSISTED_TRAINING_EVENT_TYPES = frozenset({"config", "epoch", "completed", "stopped", "error"})


class _TrainingLoss(Protocol):
    """Typed operations used from an otherwise untyped Torch loss tensor."""

    def backward(self) -> None:
        """Back-propagate the scalar loss."""

    def item(self) -> float:
        """Return the scalar loss value."""


@dataclass(frozen=True, slots=True)
class _CapturedWeightCheckpoint:
    """Hold an all-or-none serialised training checkpoint."""

    payload: bytes
    architecture: str
    parameter_count: int


class TrainingJob:
    """Manage one Studio training run for thread or process execution.

    Parameters
    ----------
    config : dict[str, Any]
        Training configuration consumed by the Studio Training Monitor.
    job_id : str or None, optional
        Stable platform job identifier. A random legacy identifier is generated
        when omitted.
    cancelled : Callable[[], bool] or None, optional
        Cooperative process-worker cancellation probe.
    event_sink : Callable[[dict[str, object]], None] or None, optional
        Sink used to persist path-free JSON events from a process worker.
    initial_state_dict : Mapping[str, object] or None, optional
        Verified model state loaded before the first optimisation step.
    """

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
        self.metrics: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._cancelled = cancelled
        self._event_sink = event_sink
        self._persisted_event_count = 0
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.final_metrics: dict[str, Any] | None = None
        self.weight_checkpoint: dict[str, JsonValue] | None = None
        self._captured_weight_checkpoint: _CapturedWeightCheckpoint | None = None
        self._initial_state_dict = initial_state_dict
        self.live_attach_evidence: dict[str, JsonValue] | None = None

    def start(self) -> None:
        """Start the legacy in-process training thread.

        The process-backed Studio route uses :meth:`run_blocking`; this method
        remains for direct callers and historical compatibility.
        """
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Request cooperative cancellation at the next training boundary."""
        self._stop_event.set()

    def run_blocking(self, context: StudioJobContext) -> dict[str, object]:
        """Run this training job inside a bounded Studio job context.

        Parameters
        ----------
        context : StudioJobContext
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
        """Queue one metric event and persist terminal-grade event classes."""
        payload: dict[str, Any] = {
            "event": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        try:
            self.metrics.put_nowait(payload)
        except queue.Full:
            with suppress(queue.Empty):
                self.metrics.get_nowait()
            self.metrics.put_nowait(payload)
        if self._event_sink is not None and event_type in _PERSISTED_TRAINING_EVENT_TYPES:
            self._event_sink(_json_event_payload(payload))
            self._persisted_event_count += 1

    def _run(self) -> None:
        """Run the legacy thread target and translate failures into events."""
        try:
            self._train()
        except Exception as exc:
            self.error = str(exc)
            self._emit("error", {"message": str(exc)})
            self.status = "failed"

    def _stop_requested(self) -> bool:
        """Return whether local or platform cancellation was requested."""
        return self._stop_event.is_set() or (self._cancelled is not None and self._cancelled())

    def _public_status(self) -> dict[str, Any]:
        """Return the path-free public status for this training job."""
        return {
            "error": self.error,
            "final_metrics": self.final_metrics,
            "job_id": self.id,
            "status": self.status,
            "weight_checkpoint": self.weight_checkpoint,
        }

    def _train(self, context: StudioJobContext | None = None) -> None:
        """Execute the Torch training loop and capture terminal weights."""
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
        learning_rate = cfg.get("lr", 1e-3)
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
        initial_state_dict = self._initial_state_dict
        if initial_state_dict is not None:
            self._attach_initial_state_dict(model, initial_state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
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
                "architecture": (
                    f"{n_inputs}→{'→'.join(str(n_hidden) for _ in range(n_layers))}→{n_outputs}"
                ),
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
                loss = cast(_TrainingLoss, spike_count_loss(spike_counts, targets))

                optimizer.zero_grad()
                loss.backward()
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
                    loss = cast(_TrainingLoss, spike_count_loss(spike_counts, targets))
                    eval_loss += loss.item() * targets.shape[0]
                    eval_correct += (spike_counts.argmax(dim=1) == targets).sum().item()
                    eval_total += targets.shape[0]

            val_loss = eval_loss / max(eval_total, 1)
            val_acc = eval_correct / max(eval_total, 1)

            layer_rates = {}
            for name in monitor.layer_names:
                raster = monitor.get(name)
                if raster is not None:
                    layer_rates[name] = float(raster.float().mean().item())

            parameter_snapshot = {}
            for parameter_name, parameter in model.named_parameters():
                if "beta_logit" in parameter_name:
                    parameter_snapshot[parameter_name] = float(
                        torch.sigmoid(parameter).mean().item()
                    )
                elif "threshold_log" in parameter_name:
                    parameter_snapshot[parameter_name] = float(torch.exp(parameter).mean().item())

            self._emit(
                "epoch",
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_accuracy": round(train_acc, 4),
                    "val_loss": round(val_loss, 6),
                    "val_accuracy": round(val_acc, 4),
                    "layer_spike_rates": layer_rates,
                    "param_snapshot": parameter_snapshot,
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

    def _attach_initial_state_dict(
        self,
        model: Any,
        state_dict: Mapping[str, object],
    ) -> None:
        """Load externally restored weights before the first optimisation step."""
        try:
            model.load_state_dict(dict(state_dict), strict=True)
        except (RuntimeError, KeyError, ValueError) as exc:
            raise ValueError(
                "Training weight attach is incompatible with the target architecture."
            ) from exc
        self._emit("attach", {"loaded_key_count": len(state_dict)})

    def _poll_live_attach(self, context: StudioJobContext, model: Any, epoch: int) -> None:
        """Consume and apply a pending live weight-attach command."""
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
        """Serialise terminal model weights for later artifact publication."""
        payload = {
            "config": self.config,
            "final_metrics": self.final_metrics,
            "model_info": model_info,
            "model_state_dict": model.state_dict(),
            "schema_version": STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
        }
        buffer = BytesIO()
        torch.save(payload, buffer)
        self._captured_weight_checkpoint = _CapturedWeightCheckpoint(
            payload=buffer.getvalue(),
            architecture=architecture,
            parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        )

    def _publish_weight_checkpoint(self, context: StudioJobContext) -> None:
        """Publish captured terminal weights into the job artifact manifest."""
        checkpoint = self._captured_weight_checkpoint
        if checkpoint is None:
            return
        self.weight_checkpoint = write_training_weight_checkpoint(
            context,
            weights_payload=checkpoint.payload,
            config=self.config,
            architecture=checkpoint.architecture,
            parameter_count=checkpoint.parameter_count,
            final_metrics=self.final_metrics,
        ).to_public_dict()


def _make_synthetic(batch_size: int) -> tuple[Any, Any, int, int]:
    """Generate synthetic classification data for quick demonstrations."""
    import torch

    n_samples = 512
    n_inputs = 64
    n_classes = 10
    features = torch.randn(n_samples, n_inputs)
    labels = torch.randint(0, n_classes, (n_samples,))
    split = int(0.8 * n_samples)
    train_dataset = TensorDataset(features[:split], labels[:split])
    test_dataset = TensorDataset(features[split:], labels[split:])
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(test_dataset, batch_size=batch_size, drop_last=True),
        n_inputs,
        n_classes,
    )


def _load_mnist(batch_size: int) -> tuple[Any, Any, int, int]:
    """Load MNIST through torchvision, or use the synthetic fallback."""
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            "~/.cache/mnist",
            train=True,
            download=True,
            transform=transform,
        )
        test_dataset = datasets.MNIST(
            "~/.cache/mnist",
            train=False,
            transform=transform,
        )
        return (
            DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(test_dataset, batch_size=batch_size),
            784,
            10,
        )
    except ImportError:
        return _make_synthetic(batch_size)
