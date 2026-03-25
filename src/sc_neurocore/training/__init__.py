# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPU SNN training with surrogate gradients

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
    from .encoding import delta_encode, latency_encode, rate_encode
    from .losses import (
        membrane_loss,
        spike_count_loss,
        spike_l1_loss,
        spike_l2_loss,
        spike_rate_loss,
    )
    from .loops import auto_device, evaluate, train_epoch
    from .utils import SpikeMonitor, model_info, population_decode, reset_states
    from .snn_modules import (
        AdExCell,
        ALIFCell,
        AlphaCell,
        ConvSpikingNet,
        ExpIFCell,
        IFCell,
        LapicqueCell,
        LIFCell,
        RecurrentLIFCell,
        SecondOrderLIFCell,
        SpikingNet,
        SynapticCell,
    )
    from .delay_linear import DelayLinear
    from .surrogate import (
        atan_surrogate,
        fast_sigmoid,
        sigmoid_surrogate,
        straight_through,
        superspike,
        triangular,
    )

__all__ = [
    "HAS_TORCH",
    # Neuron cells
    "IFCell",
    "LIFCell",
    "ALIFCell",
    "SynapticCell",
    "RecurrentLIFCell",
    "ExpIFCell",
    "AdExCell",
    "LapicqueCell",
    "AlphaCell",
    "SecondOrderLIFCell",
    "SpikingNet",
    "ConvSpikingNet",
    # Delay layer
    "DelayLinear",
    # Surrogate gradients
    "fast_sigmoid",
    "superspike",
    "atan_surrogate",
    "sigmoid_surrogate",
    "straight_through",
    "triangular",
    # Spike encoding
    "rate_encode",
    "latency_encode",
    "delta_encode",
    # Losses + regularization
    "spike_count_loss",
    "membrane_loss",
    "spike_rate_loss",
    "spike_l1_loss",
    "spike_l2_loss",
    # Training
    "auto_device",
    "train_epoch",
    "evaluate",
    # Utilities
    "SpikeMonitor",
    "model_info",
    "population_decode",
    "reset_states",
]
