# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN fine-tuning utilities

"""Layer freezing, unfreezing, and transfer learning configuration.

The helpers mutate an :class:`~sc_neurocore.transfer.checkpoint.SNNCheckpoint`
in place and return it for call chaining. Every layer reference is validated so
fine-tuning schedules cannot silently drift from the checkpoint architecture.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

from .checkpoint import SNNCheckpoint


@dataclass
class TransferConfig:
    """Configuration for checkpoint-based SNN transfer learning.

    Parameters
    ----------
    freeze_until:
        Freeze all layers up to and including this layer name or index. ``-1``
        means do not add frozen layers.
    lr_backbone:
        Learning rate for frozen backbone layers, usually zero or a small value.
    lr_head:
        Learning rate for unfrozen task-head layers.
    """

    freeze_until: str | int = -1
    lr_backbone: float = 0.0
    lr_head: float = 0.01

    def __post_init__(self) -> None:
        """Reject invalid freeze targets and learning rates."""
        if isinstance(self.freeze_until, bool) or not isinstance(self.freeze_until, (str, int)):
            raise ValueError("TransferConfig freeze_until must be a layer name or integer index")
        if isinstance(self.freeze_until, int) and self.freeze_until < -1:
            raise ValueError("TransferConfig freeze_until index must be -1 or non-negative")
        if (
            not math.isfinite(self.lr_backbone)
            or not math.isfinite(self.lr_head)
            or self.lr_backbone < 0.0
            or self.lr_head < 0.0
        ):
            raise ValueError("TransferConfig learning rates must be finite and non-negative")


def freeze_layers(
    checkpoint: SNNCheckpoint,
    layer_names: Sequence[str] | None = None,
    until_index: int | None = None,
) -> SNNCheckpoint:
    """Mark checkpoint layers as frozen.

    Parameters
    ----------
    checkpoint:
        Checkpoint to mutate.
    layer_names:
        Specific layer names to freeze.
    until_index:
        Freeze every layer with index less than or equal to this value.

    Returns
    -------
    SNNCheckpoint:
        The same checkpoint object with ``frozen_layers`` updated.
    """
    frozen = set(checkpoint.frozen_layers)

    if layer_names is not None:
        _validate_layer_names(checkpoint, layer_names)
        frozen.update(layer_names)

    if until_index is not None:
        _validate_until_index(checkpoint, until_index)
        for index, name in enumerate(checkpoint.layer_names):
            if index <= until_index:
                frozen.add(name)

    checkpoint.frozen_layers = sorted(frozen)
    return checkpoint


def unfreeze_layers(
    checkpoint: SNNCheckpoint,
    layer_names: Sequence[str] | None = None,
    all_layers: bool = False,
) -> SNNCheckpoint:
    """Mark checkpoint layers as trainable.

    Parameters
    ----------
    checkpoint:
        Checkpoint to mutate.
    layer_names:
        Specific layer names to unfreeze.
    all_layers:
        When true, clear every frozen-layer marker.

    Returns
    -------
    SNNCheckpoint:
        The same checkpoint object with ``frozen_layers`` updated.
    """
    if all_layers:
        checkpoint.frozen_layers = []
        return checkpoint

    if layer_names is not None:
        _validate_layer_names(checkpoint, layer_names)
        removals = set(layer_names)
        checkpoint.frozen_layers = [
            name for name in checkpoint.frozen_layers if name not in removals
        ]

    return checkpoint


def apply_transfer_config(
    checkpoint: SNNCheckpoint,
    config: TransferConfig,
) -> tuple[SNNCheckpoint, list[float]]:
    """Apply a transfer config and return per-layer learning rates.

    Parameters
    ----------
    checkpoint:
        Checkpoint to mutate according to ``config``.
    config:
        Validated transfer schedule.

    Returns
    -------
    tuple[SNNCheckpoint, list[float]]:
        The mutated checkpoint and one learning rate per layer.
    """
    if isinstance(config.freeze_until, int) and config.freeze_until >= 0:
        freeze_layers(checkpoint, until_index=config.freeze_until)
    elif isinstance(config.freeze_until, str):
        if config.freeze_until not in checkpoint.layer_names:
            raise ValueError("TransferConfig freeze_until layer is not present in checkpoint")
        freeze_layers(checkpoint, until_index=checkpoint.layer_names.index(config.freeze_until))

    per_layer_lr = [
        config.lr_backbone if name in checkpoint.frozen_layers else config.lr_head
        for name in checkpoint.layer_names
    ]
    return checkpoint, per_layer_lr


def _validate_layer_names(checkpoint: SNNCheckpoint, layer_names: Sequence[str]) -> None:
    if not all(isinstance(name, str) for name in layer_names):
        raise ValueError("Layer names must be strings")
    unknown = sorted(set(layer_names) - set(checkpoint.layer_names))
    if unknown:
        raise ValueError(f"Unknown layer names: {unknown}")


def _validate_until_index(checkpoint: SNNCheckpoint, until_index: int) -> None:
    if isinstance(until_index, bool) or not isinstance(until_index, int):
        raise ValueError("until_index must be an integer layer index")
    if until_index < 0 or until_index >= checkpoint.n_layers:
        raise ValueError("until_index must reference an existing layer")
