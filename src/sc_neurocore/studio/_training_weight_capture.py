# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio training weight-checkpoint capture

"""Serialising what a finished run leaves behind.

Separate from the job that supervises the run: this owns the bytes a later run
loads — the terminal weights, and the position the run reached, so a resume can
continue it rather than only begin from its network.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

from sc_neurocore.studio.platform.training_weights import (
    STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
)
from sc_neurocore.studio.training_resume import TrainingResumeState


@dataclass(frozen=True, slots=True)
class CapturedWeightCheckpoint:
    """Serialised terminal weights awaiting artifact publication.

    Attributes
    ----------
    payload : bytes
        The ``torch.save`` document, loadable with ``weights_only=True``.
    architecture : str
        Layer sizes of the network the weights belong to.
    parameter_count : int
        Total parameters, for the published metadata.
    """

    payload: bytes
    architecture: str
    parameter_count: int


def capture_weight_checkpoint(
    *,
    model: Any,
    architecture: str,
    model_info: dict[str, Any],
    config: dict[str, Any],
    final_metrics: dict[str, Any] | None,
    resume_state: TrainingResumeState | None = None,
) -> CapturedWeightCheckpoint:
    """Serialise terminal weights, and the run position when there is one.

    Parameters
    ----------
    model : torch.nn.Module
        The trained network.
    architecture : str
        Layer sizes, as the checkpoint records them.
    model_info : dict
        Architecture summary published alongside the weights.
    config : dict
        The resolved configuration the run executed.
    final_metrics : dict or None
        Terminal metrics, when the run reached them.
    resume_state : TrainingResumeState or None, optional
        The position the run reached. Present, a later run can continue this
        one exactly; absent, the weights support a warm start and nothing more.

    Returns
    -------
    CapturedWeightCheckpoint
        Bytes and metadata for artifact publication.

    Notes
    -----
    Everything written here is a tensor or a primitive, so the artifact loads
    under ``weights_only=True``. The generator states are integers and a hex
    string for exactly that reason: a checkpoint arrives from a user, and
    unpickling one would trade that guarantee for convenience.
    """
    import torch

    payload: dict[str, Any] = {
        "config": config,
        "final_metrics": final_metrics,
        "model_info": model_info,
        "model_state_dict": model.state_dict(),
        "schema_version": STUDIO_TRAINING_TORCH_STATE_DICT_SCHEMA_VERSION,
    }
    if resume_state is not None:
        payload["resume_state"] = {
            "architecture": resume_state.architecture,
            "config": dict(resume_state.config),
            "dataset_fingerprint": resume_state.dataset_fingerprint,
            "epochs_completed": resume_state.epochs_completed,
            "optimiser_state": dict(resume_state.optimiser_state),
            "rng_state": dict(resume_state.rng_state),
            "schema_version": resume_state.schema_version,
        }
    buffer = BytesIO()
    torch.save(payload, buffer)
    return CapturedWeightCheckpoint(
        payload=buffer.getvalue(),
        architecture=architecture,
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
    )


__all__ = ["CapturedWeightCheckpoint", "capture_weight_checkpoint"]
