# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Training service backend for Studio (Block 4)

from __future__ import annotations

import json
import secrets
import threading
import time
import queue
from typing import Any

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


def list_surrogates() -> list[dict[str, Any]]:
    return [{"name": s, "available": HAS_TORCH} for s in _SURROGATES]


def list_cell_types() -> list[dict[str, Any]]:
    return [{"name": c, "available": HAS_TORCH} for c in _CELL_TYPES]


class TrainingJob:
    """Manages a single training run in a background thread."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.id = f"j{secrets.token_hex(6)}"
        self.status = "pending"
        self.metrics: queue.Queue[Any] = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: str | None = None
        self.final_metrics: dict[str, Any] | None = None

    def start(self) -> None:
        self.status = "running"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

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

    def _run(self) -> None:
        try:
            self._train()
        except Exception as e:
            self.error = str(e)
            self._emit("error", {"message": str(e)})
            self.status = "failed"

    def _train(self) -> None:
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
            if self._stop_event.is_set():
                self.status = "stopped"
                self._emit("stopped", {"epoch": epoch})
                return

            model.train()
            monitor.reset()
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, targets) in enumerate(train_loader):
                if self._stop_event.is_set():
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
        self._emit("completed", self.final_metrics)
        monitor.remove()


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


def start_training(config: dict[str, Any]) -> dict[str, Any]:
    job = TrainingJob(config)
    with _jobs_lock:
        _jobs[job.id] = job
    job.start()
    return {"job_id": job.id, "status": "running"}


def stop_training(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    job.stop()
    return {"job_id": job_id, "status": "stopping"}


def get_training_status(job_id: str) -> dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return {"error": f"Job {job_id} not found"}
    return {
        "job_id": job.id,
        "status": job.status,
        "error": job.error,
        "final_metrics": job.final_metrics,
    }


def stream_metrics(job_id: str) -> Any:
    """Generator that yields SSE-formatted metric events."""
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'event': 'error', 'data': {'message': 'Job not found'}})}\n\n"
        return

    while True:
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
    with _jobs_lock:
        return [{"job_id": j.id, "status": j.status, "config": j.config} for j in _jobs.values()]
