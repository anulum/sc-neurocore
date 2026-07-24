# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_federated_sc.py

from __future__ import annotations

import numpy as np
from sc_neurocore.federated.federated_sc import (
    AdaptiveEpsilonScheduler,
    AuditLog,
    CommitmentScheme,
    ConvergenceTracker,
    DPCertificate,
    DPMechanism,
    ErrorFeedback,
    FederatedAggregator,
    FederatedClient,
    FederatedRound,
    PrivacyAccountant,
    SCGradientEncoder,
    SecretShare,
    amplified_epsilon,
    bitstream_probability,
    clip_gradients,
    fedprox_gradient,
    krum_select,
    lfsr_encode,
    poisson_subsample,
    sparsify_topk,
    stochastic_quantize,
    trimmed_mean,
)

__all__ = [
    "np",
    "AdaptiveEpsilonScheduler",
    "AuditLog",
    "CommitmentScheme",
    "ConvergenceTracker",
    "DPCertificate",
    "DPMechanism",
    "ErrorFeedback",
    "FederatedAggregator",
    "FederatedClient",
    "FederatedRound",
    "PrivacyAccountant",
    "SCGradientEncoder",
    "SecretShare",
    "amplified_epsilon",
    "bitstream_probability",
    "clip_gradients",
    "fedprox_gradient",
    "krum_select",
    "lfsr_encode",
    "poisson_subsample",
    "sparsify_topk",
    "stochastic_quantize",
    "trimmed_mean",
]
