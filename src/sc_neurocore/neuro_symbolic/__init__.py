# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.neuro_symbolic -- Predictive coding and hyperdimensional symbol binding

"""sc_neurocore.neuro_symbolic -- Predictive coding and hyperdimensional symbol binding.

Tier: experimental.
"""

__tier__ = "experimental"

from .agent import (
    HybridInferenceResult,
    NeuroSymbolicPredictiveAgent,
    PredictiveAgentConfig,
    SCErrorSignature,
    build_sc_error_signature,
)
from .predictive_coding import (
    Hypervector,
    PredictiveCodingLayer,
    ReasoningTrace,
    SymbolEncoder,
    VerifiableInference,
)
from .self_verification import (
    NeuroSymbolicSelfVerificationTrace,
    NeuroSymbolicSelfVerifier,
    VerificationObligation,
    VerificationStatus,
    build_self_verification_trace,
)

__all__ = [
    "HybridInferenceResult",
    "NeuroSymbolicPredictiveAgent",
    "PredictiveAgentConfig",
    "SCErrorSignature",
    "build_sc_error_signature",
    "Hypervector",
    "PredictiveCodingLayer",
    "ReasoningTrace",
    "SymbolEncoder",
    "VerifiableInference",
    "NeuroSymbolicSelfVerificationTrace",
    "NeuroSymbolicSelfVerifier",
    "VerificationObligation",
    "VerificationStatus",
    "build_self_verification_trace",
]
