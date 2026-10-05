# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio QCFS conversion training route

"""Train a QCFS ANN, convert it and judge the converted network, not the source.

The source is a dense classifier whose hidden activations are QCFS quantisers
with the run's ``timesteps`` as their step budget. After the last epoch it is
converted to a dense integrate-and-fire network with that same budget, and
both networks classify the validation split with constant-current input. The
run's ``val_accuracy`` is the converted network's; the source's accuracy and
the difference are reported beside it, and the measured comparison is sealed
as ``training/conversion_report.json``. A run that names a ``target_profile``
also fits the converted network into that profile's fixed-point format, replays
the rounded network on the same samples and seals ``training/target_report.json``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from sc_neurocore.studio._training_datasets import _load_mnist, _make_synthetic
from sc_neurocore.studio.training_refusals import TrainingRefusal

#: Artifact holding the measured source-to-converted comparison.
CONVERSION_REPORT_ARTIFACT_PATH = "training/conversion_report.json"

#: Artifact holding the converted network's fit into a requested target format.
TARGET_REPORT_ARTIFACT_PATH = "training/target_report.json"


@dataclass(frozen=True, slots=True)
class ConversionOutcome:
    """A finished conversion run, ready to be recorded by its job.

    Attributes
    ----------
    model : torch.nn.Module
        The trained QCFS source network.
    architecture : str
        Layer sizes, input to output.
    model_info : dict
        Parameter counts and cell summary of the source.
    observed : dict
        Unrounded terminal metrics; the job rounds them for publication and
        judges a preregistered criterion on them as they are.
    report : dict
        The sealed conversion loss report.
    target_report : dict or None
        The sealed target calibration, when a target profile was named.
    """

    model: Any
    architecture: str
    model_info: dict[str, Any]
    observed: dict[str, float]
    report: dict[str, Any]
    target_report: dict[str, Any] | None = None


def _evaluate(model: Any, loader: Any, device: Any) -> tuple[float, float, Any, Any]:
    """Return source loss and accuracy on a loader, with the samples it served.

    Raises
    ------
    TrainingRefusal
        The loader served no samples, so there is nothing to judge a conversion on.
    """
    import torch

    model.eval()
    loss_sum, correct, inputs, labels = 0.0, 0, [], []
    with torch.no_grad():
        for data, targets in loader:
            logits = model(data.to(device))
            targets = targets.to(device)
            loss_sum += float(torch.nn.functional.cross_entropy(logits, targets)) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            inputs.append(data)
            labels.append(targets.cpu())
    if not labels:
        raise TrainingRefusal(
            "The validation split served no samples at this batch size; a conversion "
            "cannot be judged on none. Choose a smaller batch size."
        )
    total = sum(len(batch) for batch in labels)
    return loss_sum / total, correct / total, torch.cat(inputs), torch.cat(labels)


def train_qcfs_conversion(
    resolved: Any,
    context: Any,
    *,
    job_id: str,
    emit: Callable[[str, dict[str, Any]], None],
    stop_requested: Callable[[], bool],
) -> ConversionOutcome | None:
    """Run the conversion route to completion, or stop at a batch boundary.

    Parameters
    ----------
    resolved : ResolvedTrainingConfig
        A configuration whose ``model_kind`` is ``qcfs_conversion``.
    context : StudioJobContext or None
        Job sandbox the conversion report is sealed into; ``None`` for a
        direct in-process run, which keeps the report on the outcome only.
    job_id : str
        Identifier published in the configuration event.
    emit : callable
        Event sink of the supervising job.
    stop_requested : callable
        Cooperative cancellation probe.

    Returns
    -------
    ConversionOutcome or None
        The finished run, or ``None`` after a ``stopped`` event.
    """
    import torch

    from sc_neurocore.conversion import convert, measure_conversion_loss
    from sc_neurocore.conversion.checkpoint_network import build_qcfs_classifier
    from sc_neurocore.studio.training_resume import dataset_fingerprint
    from sc_neurocore.training import auto_device, model_info

    loader = _load_mnist if resolved.dataset == "mnist" else _make_synthetic
    train_loader, test_loader, n_inputs, n_outputs = loader(resolved.batch_size, rates=True)
    device = auto_device()
    model = build_qcfs_classifier(n_inputs, resolved.hidden_widths, n_outputs, resolved.timesteps)
    model = model.to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=resolved.learning_rate)
    architecture = resolved.architecture(n_inputs, n_outputs)
    info = model_info(model)
    emit(
        "config",
        {
            "job_id": job_id,
            "device": str(device),
            "model_info": info,
            "model_kind": resolved.model_kind,
            "dataset": resolved.dataset,
            "n_epochs": resolved.epochs,
            "architecture": architecture,
            "resolved_config": resolved.to_public_dict(),
            "dataset_fingerprint": dataset_fingerprint(train_loader),
            "input_data": None,
            "input_admission": None,
            "start_epoch": 0,
        },
    )
    for epoch in range(resolved.epochs):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for batch, (data, targets) in enumerate(train_loader):
            if stop_requested():
                emit("stopped", {"epoch": epoch})
                return None
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = torch.nn.functional.cross_entropy(logits, targets)
            optimiser.zero_grad()
            torch.autograd.backward(loss)
            if resolved.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), resolved.max_grad_norm)
            optimiser.step()
            loss_sum += float(loss.detach()) * len(targets)
            correct += int((logits.argmax(1) == targets).sum())
            total += len(targets)
            if (batch + 1) % 10 == 0:
                emit(
                    "batch",
                    {
                        "epoch": epoch,
                        "batch": batch + 1,
                        "loss": float(loss.detach()),
                        "accuracy": correct / total,
                    },
                )
        train_loss, train_accuracy = loss_sum / max(total, 1), correct / max(total, 1)
        val_loss, val_accuracy, inputs, labels = _evaluate(model, test_loader, device)
        emit(
            "epoch",
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_accuracy": round(train_accuracy, 4),
                "val_loss": round(val_loss, 6),
                "val_accuracy": round(val_accuracy, 4),
                "layer_spike_rates": {},
                "param_snapshot": {
                    name: float(parameter.detach())
                    for name, parameter in model.named_parameters()
                    if name.endswith("theta")
                },
            },
        )
    if stop_requested():
        emit("stopped", {"epoch": resolved.epochs})
        return None
    snn = convert(model, T=resolved.timesteps)
    report = measure_conversion_loss(
        model,
        snn,
        inputs.numpy(),
        labels.numpy(),
        input_mode="constant",
        batch_size=resolved.batch_size,
    ).to_public_dict()
    target = _calibrate(resolved, snn, inputs, labels)
    if context is not None:
        context.write_artifact(CONVERSION_REPORT_ARTIFACT_PATH, json.dumps(report, sort_keys=True))
        if target is not None:
            context.write_artifact(TARGET_REPORT_ARTIFACT_PATH, json.dumps(target, sort_keys=True))
    observed = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "val_loss": val_loss,
        "val_accuracy": report["converted_accuracy"],
        "source_val_accuracy": report["source_accuracy"],
        "conversion_accuracy_drop": report["accuracy_drop"],
    }
    if target is not None:
        observed["target_accuracy"] = target["quantized_accuracy"]
    return ConversionOutcome(model, architecture, info, observed, report, target)


def _calibrate(resolved: Any, snn: Any, inputs: Any, labels: Any) -> dict[str, Any] | None:
    """Fit the converted network into the requested target format, when one was named."""
    if resolved.target_profile is None:
        return None
    from sc_neurocore.compiler.platforms import get_profile
    from sc_neurocore.conversion.target_report import calibrate_for_target

    return calibrate_for_target(
        snn,
        get_profile(resolved.target_profile),
        inputs.reshape(len(inputs), -1).numpy(),
        labels.numpy(),
        batch_size=resolved.batch_size,
    ).to_public_dict()


__all__ = [
    "CONVERSION_REPORT_ARTIFACT_PATH",
    "TARGET_REPORT_ARTIFACT_PATH",
    "ConversionOutcome",
    "train_qcfs_conversion",
]
