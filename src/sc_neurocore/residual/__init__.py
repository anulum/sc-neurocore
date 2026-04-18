# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SNN residual blocks for deep spiking networks

"""SNN residual learning: MS-ResNet, SEW, pre-activation blocks. 400+ layer SNNs."""

from .blocks import MembraneShortcutBlock, SEWBlock, DeepSNNStack

__all__ = ["MembraneShortcutBlock", "SEWBlock", "DeepSNNStack"]
