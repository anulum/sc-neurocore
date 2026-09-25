# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training job execution

"""Execute one Studio SNN training run and publish its bounded artifacts."""

from __future__ import annotations

import importlib.util
import json
import queue
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import suppress
from typing import Any, Protocol, cast

from sc_neurocore.studio._training_datasets import _load_mnist, _make_synthetic, _seed_everything
from sc_neurocore.studio._training_weight_capture import (
    CapturedWeightCheckpoint,
    capture_weight_checkpoint,
)
from sc_neurocore.studio.training_resume import (
    TrainingResumeState,
    apply_resume_state,
    capture_resume_state,
    dataset_fingerprint,
)
from sc_neurocore.studio.training_contract import (
    SUPPORTED_CELL_TYPES,
    SUPPORTED_SURROGATES,
    resolve_training_config,
)
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
    write_training_weight_checkpoint,
)

# Whether PyTorch is installed, found without importing it: every process that
# imports the Studio routers imports this module, and importing Torch there
# starts its thread pools and maps its libraries. An analysis worker runs under
# an address-space limit it would then spend on a library it never calls.
HAS_TORCH = importlib.util.find_spec("torch") is not None


# One owner for the supported vocabularies: the contract that refuses an
# unsupported choice is the same list the capability routes advertise, so the
# Studio cannot offer a name its runner would reject.
_SURROGATES = SUPPORTED_SURROGATES
_CELL_TYPES = SUPPORTED_CELL_TYPES

_PERSISTED_TRAINING_EVENT_TYPES = frozenset({"config", "epoch", "completed", "stopped", "error"})

# Python, NumPy and Torch generators are process-global. Two training runs in
# one process that draw from them at the same time disturb each other, and
# neither can then be replayed or resumed exactly, so a process trains one job
# at a time. The process-backed Studio route runs each job in its own process
# and never waits here; the legacy in-process thread and direct callers do.
_GLOBAL_GENERATORS = threading.Lock()


class _TrainingLoss(Protocol):
    """Typed operations used from an otherwise untyped Torch loss tensor."""

    def backward(self) -> None:
        """Back-propagate the scalar loss."""

    def item(self) -> float:
        """Return the scalar loss value."""


class TrainingJob:
    """Manage one Studio training run for thread or process execution.

    Parameters
    ----------
    config : dict[str, Any]
        Training request. It is resolved against the training contract here, so
        an unsupported dataset, surrogate or layer width is refused before the
        job exists rather than part-way through a run.
    job_id : str or None, optional
        Stable platform job identifier. A random legacy identifier is generated
        when omitted.
    cancelled : Callable[[], bool] or None, optional
        Cooperative process-worker cancellation probe.
    event_sink : Callable[[dict[str, object]], None] or None, optional
        Sink used to persist path-free JSON events from a process worker.
    initial_state_dict : Mapping[str, object] or None, optional
        Verified model state loaded before the first optimisation step. On its
        own this is a **warm start**: a new run beginning from those weights,
        with a fresh optimiser and generator at epoch zero.
    resume_state : TrainingResumeState or None, optional
        Saved position of a run to continue **exactly**: the optimiser state,
        the generator states and how many epochs are already done. Supplied
        together with ``initial_state_dict``; supplying it alone would restore
        an optimiser onto weights it never saw.

    Raises
    ------
    TrainingConfigError
        The request names something the Studio cannot run.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        job_id: str | None = None,
        cancelled: Callable[[], bool] | None = None,
        event_sink: Callable[[dict[str, object]], None] | None = None,
        initial_state_dict: Mapping[str, object] | None = None,
        resume_state: TrainingResumeState | None = None,
    ) -> None:
        # Resolved here so an unrunnable job cannot be constructed at all: a
        # refusal after the artifact machinery has started produces a second,
        # confusing error about a missing artifact instead of the real one.
        self._resume_state = resume_state
        self.resolved_config = resolve_training_config(config)
        self.config = dict(self.resolved_config.to_public_dict())
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
        self._captured_weight_checkpoint: CapturedWeightCheckpoint | None = None
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

    @staticmethod
    def write_refused_evidence(context: StudioJobContext, message: str) -> None:
        """Seal failed evidence for a request that was refused before it ran.

        Parameters
        ----------
        context : StudioJobContext
            Job sandbox to write the status and evidence artifacts into.
        message : str
            The refusal, as the caller will read it.

        Notes
        -----
        A refused configuration never builds a model, so there is no weight
        checkpoint and no event log to publish — only the reason. Writing it
        keeps the sandbox's account complete: every job that ends has an
        evidence artifact saying how.
        """
        status_payload: dict[str, object] = {
            "job_id": context.job_id,
            "status": "failed",
            "error": message,
            "final_metrics": None,
            "weight_checkpoint": None,
        }
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
            status="failed",
            error_message=message,
        )

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

    def _finish_if_stopped(self, epoch: int, monitor: Any) -> bool:
        """Close monitor hooks and report a stop at a training boundary."""
        if not self._stop_requested():
            return False
        self.status = "stopped"
        self._emit("stopped", {"epoch": epoch})
        monitor.remove()
        return True

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
        """Train while holding this process's global generators."""
        with _GLOBAL_GENERATORS:
            self._train_seeded(context)

    def _train_seeded(self, context: StudioJobContext | None) -> None:
        """Execute the Torch training loop and capture terminal weights."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not installed. pip install sc-neurocore[research]")

        import torch

        from sc_neurocore.training import (
            SpikingNet,
            SpikeMonitor,
            auto_device,
            model_info,
            spike_count_loss,
        )
        from sc_neurocore.training import surrogate as surr_mod

        resolved = self.resolved_config
        dataset = resolved.dataset
        n_epochs = resolved.epochs
        batch_size = resolved.batch_size
        learning_rate = resolved.learning_rate
        n_timesteps = resolved.timesteps
        surrogate_name = resolved.surrogate
        max_grad_norm = resolved.max_grad_norm

        # Seeded before the loaders and the model, because both draw from the
        # global generators: an unseeded run cannot be replayed and its
        # checkpoint records a result nobody can reproduce.
        _seed_everything(resolved.seed)

        surrogate_fn = getattr(surr_mod, surrogate_name)
        device = auto_device()

        if dataset == "mnist":
            train_loader, test_loader, n_inputs, n_outputs = _load_mnist(batch_size)
        else:
            train_loader, test_loader, n_inputs, n_outputs = _make_synthetic(batch_size)

        train_fingerprint = dataset_fingerprint(train_loader)
        model = SpikingNet(
            n_input=n_inputs,
            n_hidden=list(resolved.hidden_widths),
            n_output=n_outputs,
            n_layers=len(resolved.hidden_widths),
            surrogate_fn=surrogate_fn,
            learn_beta=resolved.learn_beta,
            learn_threshold=resolved.learn_threshold,
        ).to(device)
        initial_state_dict = self._initial_state_dict
        if initial_state_dict is not None:
            self._attach_initial_state_dict(model, initial_state_dict)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        architecture = resolved.architecture(n_inputs, n_outputs)
        start_epoch = 0
        if self._resume_state is not None:
            # Restores the optimiser and the generator states, and returns the
            # epoch this run begins at, so a resumed run neither repeats nor
            # skips the work the interrupted one already did.
            start_epoch = apply_resume_state(
                self._resume_state,
                optimiser=optimizer,
                architecture=architecture,
                config=resolved.to_public_dict(),
            )
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
                "architecture": architecture,
                "resolved_config": resolved.to_public_dict(),
                "dataset_fingerprint": train_fingerprint,
                "start_epoch": start_epoch,
            },
        )

        for epoch in range(start_epoch, n_epochs):
            if self._finish_if_stopped(epoch, monitor):
                return

            if context is not None:
                self._poll_live_attach(context, model, epoch)

            model.train()
            monitor.reset()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                if self._finish_if_stopped(epoch, monitor):
                    return

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

            if self._finish_if_stopped(epoch, monitor):
                return

            train_loss = epoch_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            model.eval()
            eval_loss = 0.0
            eval_correct = 0
            eval_total = 0
            with torch.no_grad():
                for data, targets in test_loader:
                    if self._finish_if_stopped(epoch, monitor):
                        return
                    data, targets = data.to(device), targets.to(device)
                    data = data.view(data.shape[0], -1)
                    data = data.unsqueeze(0).expand(n_timesteps, *data.shape)
                    spike_counts, _ = model(data)
                    loss = cast(_TrainingLoss, spike_count_loss(spike_counts, targets))
                    eval_loss += loss.item() * targets.shape[0]
                    eval_correct += (spike_counts.argmax(dim=1) == targets).sum().item()
                    eval_total += targets.shape[0]

            if self._finish_if_stopped(epoch, monitor):
                return

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

        if self._finish_if_stopped(n_epochs, monitor):
            return
        self.status = "completed"
        self.final_metrics = {
            "train_loss": round(train_loss, 6),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 6),
            "val_accuracy": round(val_acc, 4),
        }
        self._capture_weight_checkpoint(
            model=model,
            architecture=architecture,
            model_info=info,
            resume_state=capture_resume_state(
                epochs_completed=n_epochs,
                architecture=architecture,
                config=resolved.to_public_dict(),
                optimiser=optimizer,
                dataset_fingerprint=train_fingerprint,
            ),
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
        resume_state: TrainingResumeState | None = None,
    ) -> None:
        """Serialise terminal weights and the position of the run."""
        self._captured_weight_checkpoint = capture_weight_checkpoint(
            model=model,
            architecture=architecture,
            model_info=model_info,
            config=self.config,
            final_metrics=self.final_metrics,
            resume_state=resume_state,
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
