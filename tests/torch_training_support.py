# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_torch_training.py

from __future__ import annotations

import pytest
torch = pytest.importorskip("torch")
import sc_neurocore.training.loops as training_loops
from sc_neurocore.training import (
    HAS_TORCH,
    # Cells
    IFCell,
    LIFCell,
    ALIFCell,
    SynapticCell,
    RecurrentLIFCell,
    ExpIFCell,
    AdExCell,
    LapicqueCell,
    AlphaCell,
    SecondOrderLIFCell,
    # Networks
    SpikingNet,
    ConvSpikingNet,
    # Surrogate
    fast_sigmoid,
    superspike,
    atan_surrogate,
    sigmoid_surrogate,
    straight_through,
    triangular,
    # Encoding
    rate_encode,
    latency_encode,
    delta_encode,
    # Losses
    spike_count_loss,
    membrane_loss,
    spike_rate_loss,
    spike_l1_loss,
    spike_l2_loss,
    # Training
    auto_device,
    train_epoch,
    evaluate,
    # Utilities
    SpikeMonitor,
    model_info,
    population_decode,
    reset_states,
    # Delay
    DelayLinear,
)

__all__ = ['pytest', 'torch', 'training_loops', 'HAS_TORCH', 'IFCell', 'LIFCell', 'ALIFCell', 'SynapticCell', 'RecurrentLIFCell', 'ExpIFCell', 'AdExCell', 'LapicqueCell', 'AlphaCell', 'SecondOrderLIFCell', 'SpikingNet', 'ConvSpikingNet', 'fast_sigmoid', 'superspike', 'atan_surrogate', 'sigmoid_surrogate', 'straight_through', 'triangular', 'rate_encode', 'latency_encode', 'delta_encode', 'spike_count_loss', 'membrane_loss', 'spike_rate_loss', 'spike_l1_loss', 'spike_l2_loss', 'auto_device', 'train_epoch', 'evaluate', 'SpikeMonitor', 'model_info', 'population_decode', 'reset_states', 'DelayLinear', '__all__']
