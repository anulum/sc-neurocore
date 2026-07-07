# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.core -- Tier: research (experimental /

"""sc_neurocore.core -- Tier: research (experimental / research)."""

__tier__ = "research"

from .bipolar import (
    bipolar_decode,
    bipolar_encode,
    bipolar_mac,
    bipolar_multiply,
    bipolar_sc_layer,
    float_to_bipolar_weights,
)
from .mdl_parser import MindDescriptionLanguage, MDLSpecification
from .orchestrator import CognitiveOrchestrator
from .sc_correlation import (
    CorrelationDiagnostic,
    correlation_diagnostic,
    estimate_scc,
    observed_and_bias,
)
from .sc_error_bounds import (
    SCErrorBound,
    bernoulli_std_error,
    bernoulli_variance,
    bipolar_std_error,
    bipolar_variance,
    dot_product_variance,
    hoeffding_confidence,
    hoeffding_min_length,
    low_discrepancy_error_bound,
    min_length_for_std_error,
    multiply_correlation_bias,
    multiply_variance,
    mux_add_variance,
    sc_error_bound,
)
from .tensor_stream import TensorStream

__all__ = [
    "bipolar_decode",
    "bipolar_encode",
    "bipolar_mac",
    "bipolar_multiply",
    "bipolar_sc_layer",
    "float_to_bipolar_weights",
    "MindDescriptionLanguage",
    "MDLSpecification",
    "CognitiveOrchestrator",
    "CorrelationDiagnostic",
    "estimate_scc",
    "observed_and_bias",
    "correlation_diagnostic",
    "SCErrorBound",
    "bernoulli_variance",
    "bernoulli_std_error",
    "bipolar_variance",
    "bipolar_std_error",
    "multiply_variance",
    "multiply_correlation_bias",
    "mux_add_variance",
    "dot_product_variance",
    "low_discrepancy_error_bound",
    "hoeffding_confidence",
    "hoeffding_min_length",
    "min_length_for_std_error",
    "sc_error_bound",
    "TensorStream",
]
