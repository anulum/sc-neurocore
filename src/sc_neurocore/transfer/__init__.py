# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN transfer learning + checkpoint serialization

"""Checkpoint serialization and transfer-learning helpers for SNN models.

The package exports the complete public transfer workflow: build or load a
validated checkpoint, freeze or unfreeze named layers, apply a learning-rate
schedule, and save the resulting state back to disk.
"""

from .checkpoint import SNNCheckpoint, load_checkpoint, save_checkpoint
from .fine_tune import TransferConfig, apply_transfer_config, freeze_layers, unfreeze_layers

__all__ = [
    "SNNCheckpoint",
    "TransferConfig",
    "apply_transfer_config",
    "freeze_layers",
    "save_checkpoint",
    "load_checkpoint",
    "unfreeze_layers",
]
