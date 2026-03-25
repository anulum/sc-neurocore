# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN fine-tuning utilities

"""Layer freezing, unfreezing, and transfer learning configuration.

Pretrain on large dataset → freeze backbone → fine-tune head on small dataset.
The foundation of modern ML that SNNs completely lacked.
"""

from __future__ import annotations

from dataclasses import dataclass


from .checkpoint import SNNCheckpoint


@dataclass
class TransferConfig:
    """Configuration for transfer learning.

    Parameters
    ----------
    freeze_until : str or int
        Freeze all layers up to (and including) this layer name or index.
    lr_backbone : float
        Learning rate for frozen layers (usually 0 or very small).
    lr_head : float
        Learning rate for unfrozen layers.
    """

    freeze_until: str | int = -1
    lr_backbone: float = 0.0
    lr_head: float = 0.01


def freeze_layers(
    checkpoint: SNNCheckpoint,
    layer_names: list[str] | None = None,
    until_index: int | None = None,
) -> SNNCheckpoint:
    """Freeze layers (mark as non-trainable).

    Parameters
    ----------
    checkpoint : SNNCheckpoint
    layer_names : list of str, optional
        Specific layers to freeze.
    until_index : int, optional
        Freeze all layers with index <= until_index.

    Returns
    -------
    SNNCheckpoint with frozen_layers updated
    """
    frozen = set(checkpoint.frozen_layers)

    if layer_names is not None:
        frozen.update(layer_names)

    if until_index is not None:
        for i, name in enumerate(checkpoint.layer_names):
            if i <= until_index:
                frozen.add(name)

    checkpoint.frozen_layers = sorted(frozen)
    return checkpoint


def unfreeze_layers(
    checkpoint: SNNCheckpoint,
    layer_names: list[str] | None = None,
    all_layers: bool = False,
) -> SNNCheckpoint:
    """Unfreeze layers (mark as trainable).

    Parameters
    ----------
    checkpoint : SNNCheckpoint
    layer_names : list of str, optional
        Specific layers to unfreeze.
    all_layers : bool
        If True, unfreeze everything.
    """
    if all_layers:
        checkpoint.frozen_layers = []
        return checkpoint

    if layer_names is not None:
        checkpoint.frozen_layers = [n for n in checkpoint.frozen_layers if n not in layer_names]

    return checkpoint


def apply_transfer_config(
    checkpoint: SNNCheckpoint,
    config: TransferConfig,
) -> tuple[SNNCheckpoint, list[float]]:
    """Apply transfer config: freeze layers, compute per-layer LR.

    Returns
    -------
    (checkpoint, per_layer_lr)
    """
    if isinstance(config.freeze_until, int) and config.freeze_until >= 0:
        freeze_layers(checkpoint, until_index=config.freeze_until)
    elif isinstance(config.freeze_until, str):
        idx = (
            checkpoint.layer_names.index(config.freeze_until)
            if config.freeze_until in checkpoint.layer_names
            else -1
        )
        if idx >= 0:
            freeze_layers(checkpoint, until_index=idx)

    per_layer_lr = []
    for name in checkpoint.layer_names:
        if name in checkpoint.frozen_layers:
            per_layer_lr.append(config.lr_backbone)
        else:
            per_layer_lr.append(config.lr_head)

    return checkpoint, per_layer_lr
