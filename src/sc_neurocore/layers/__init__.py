# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layers Package Init

"""Expose stochastic-computing layer primitives for package consumers."""

from .sc_dense_layer import SCDenseLayer
from .sc_conv_layer import SCConv2DLayer
from .sc_learning_layer import SCLearningLayer
from .vectorized_layer import VectorizedSCLayer
from .recurrent import SCRecurrentLayer
from .memristive import MemristiveDenseLayer
from .fusion import SCFusionLayer
from .attention import StochasticAttention
from .hardware_aware import HardwareAwareSCLayer
from .predictive_coding import PredictiveCodingSCLayer
from .rall_dendrite import RallDendrite
from .circuit_primitives import LateralInhibition, WinnerTakeAll

__all__ = [
    "SCDenseLayer",
    "SCConv2DLayer",
    "SCLearningLayer",
    "VectorizedSCLayer",
    "SCRecurrentLayer",
    "MemristiveDenseLayer",
    "SCFusionLayer",
    "StochasticAttention",
    "HardwareAwareSCLayer",
    "PredictiveCodingSCLayer",
    "RallDendrite",
    "LateralInhibition",
    "WinnerTakeAll",
]
