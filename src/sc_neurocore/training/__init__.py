# SPDX-License-Identifier: AGPL-3.0-or-later
"""GPU SNN training with surrogate gradients.

Requires PyTorch: pip install sc-neurocore[research]
"""

from __future__ import annotations

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .losses import membrane_loss, spike_count_loss, spike_rate_loss
    from .loops import evaluate, train_epoch
    from .snn_modules import ALIFCell, ConvSpikingNet, LIFCell, RecurrentLIFCell, SpikingNet
    from .surrogate import atan_surrogate, fast_sigmoid, superspike

__all__ = [
    "HAS_TORCH",
    "ALIFCell",
    "ConvSpikingNet",
    "LIFCell",
    "RecurrentLIFCell",
    "SpikingNet",
    "fast_sigmoid",
    "superspike",
    "atan_surrogate",
    "spike_count_loss",
    "membrane_loss",
    "spike_rate_loss",
    "train_epoch",
    "evaluate",
]
