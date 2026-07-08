# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — GPU SNN training with surrogate gradients

"""GPU SNN training with surrogate gradients.

Requires PyTorch: pip install sc-neurocore[research]
"""

from __future__ import annotations

from .equilibrium_propagation import EPNetwork

try:
    import torch as _torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

if HAS_TORCH:
    from .delay_linear import DelayLinear
    from .encoding import delta_encode, latency_encode, rate_encode
    from .loops import auto_device, evaluate, train_epoch
    from .losses import (
        membrane_loss,
        spike_count_loss,
        spike_l1_loss,
        spike_l2_loss,
        spike_rate_loss,
    )
    from .sc_correlation_regularizers import (
        correlation_matrix,
        correlation_penalty,
        pairwise_correlation_penalty,
    )
    from .sc_estimators import (
        DifferentiableSCConfig,
        RelaxedSCProduct,
        SampledSCProduct,
        SCBitstreamSample,
        SCBitstreamStatistics,
        estimate_bitstream_statistics,
        finite_difference_gradients,
        relaxed_sc_multiply,
        sample_sc_bitstreams,
        sampled_sc_multiply,
    )
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
        SCWeightNoiseModel,
        SecondOrderLIFCell,
        SpikingNet,
        SynapticCell,
    )
    from .stochastic_backprop import (
        SCBackpropDesignSpace,
        SCBackpropJointReport,
        SCObjectiveBreakdown,
        SCResourceProxy,
        SCTrainingObjectiveConfig,
        relaxed_sc_linear,
        stochastic_backprop_joint_objective,
        stochastic_training_objective,
    )
    from .stochastic_backprop_export import (
        STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION,
        build_stochastic_backprop_export_manifest,
        write_stochastic_backprop_export_manifest,
        write_stochastic_backprop_handoff_bundle,
    )
    from .surrogate import (
        SURROGATE_PATHS,
        atan_surrogate,
        atan_surrogate_custom_op,
        atan_surrogate_legacy,
        fast_sigmoid,
        fast_sigmoid_custom_op,
        fast_sigmoid_legacy,
        sigmoid_surrogate,
        sigmoid_surrogate_custom_op,
        sigmoid_surrogate_legacy,
        straight_through,
        straight_through_custom_op,
        straight_through_legacy,
        superspike,
        superspike_custom_op,
        superspike_legacy,
        triangular,
        triangular_custom_op,
        triangular_legacy,
    )
    from .utils import SpikeMonitor, model_info, population_decode, reset_states

__all__ = [
    "HAS_TORCH",
    # Equilibrium propagation
    "EPNetwork",
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
    "SCWeightNoiseModel",
    # Delay layer
    "DelayLinear",
    # Surrogate gradients
    "SURROGATE_PATHS",
    "fast_sigmoid",
    "fast_sigmoid_custom_op",
    "fast_sigmoid_legacy",
    "superspike",
    "superspike_custom_op",
    "superspike_legacy",
    "atan_surrogate",
    "atan_surrogate_custom_op",
    "atan_surrogate_legacy",
    "sigmoid_surrogate",
    "sigmoid_surrogate_custom_op",
    "sigmoid_surrogate_legacy",
    "straight_through",
    "straight_through_custom_op",
    "straight_through_legacy",
    "triangular",
    "triangular_custom_op",
    "triangular_legacy",
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
    # Differentiable stochastic-computing estimators
    "DifferentiableSCConfig",
    "RelaxedSCProduct",
    "SCBitstreamSample",
    "SCBitstreamStatistics",
    "SampledSCProduct",
    "relaxed_sc_multiply",
    "sample_sc_bitstreams",
    "sampled_sc_multiply",
    "estimate_bitstream_statistics",
    "finite_difference_gradients",
    "correlation_matrix",
    "correlation_penalty",
    "pairwise_correlation_penalty",
    "SCObjectiveBreakdown",
    "SCBackpropDesignSpace",
    "SCBackpropJointReport",
    "SCResourceProxy",
    "SCTrainingObjectiveConfig",
    "relaxed_sc_linear",
    "stochastic_backprop_joint_objective",
    "stochastic_training_objective",
    "STOCHASTIC_BACKPROP_EXPORT_SCHEMA_VERSION",
    "build_stochastic_backprop_export_manifest",
    "write_stochastic_backprop_handoff_bundle",
    "write_stochastic_backprop_export_manifest",
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
