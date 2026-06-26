# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — sc_neurocore.learning -- Tier: research (experimental /

"""sc_neurocore.learning -- Tier: research (experimental / research)."""

__tier__ = "research"

from .advanced import (
    BPTTLearner,
    TBPTTLearner,
    EligibilityTrace,
    HomeostaticPlasticity,
    MetaLearner,
    RewardModulatedLearner,
    ShortTermPlasticity,
    StructuralPlasticity,
)
from .callbacks import CSVCallback, TensorBoardCallback, TrainingCallback, WandBCallback
from .federated import FederatedAggregator
from .lifelong import EWC_SCLayer
from .neuroevolution import SNNGeneticEvolver
from .online_o1 import (
    ONLINE_O1_ANNOTATION_SCHEMA_VERSION,
    ONLINE_O1_MEMORY_PROOF_SCHEMA_VERSION,
    OnlineO1Config,
    OnlineO1Snapshot,
    OnlineO1Synapse,
    build_online_o1_memory_proof,
)
from .schedulers import (
    CosineScheduler,
    ExponentialScheduler,
    StepScheduler,
    WarmupCosineScheduler,
)

__all__ = [
    "BPTTLearner",
    "CSVCallback",
    "CosineScheduler",
    "ExponentialScheduler",
    "TBPTTLearner",
    "EligibilityTrace",
    "FederatedAggregator",
    "EWC_SCLayer",
    "HomeostaticPlasticity",
    "MetaLearner",
    "ONLINE_O1_ANNOTATION_SCHEMA_VERSION",
    "ONLINE_O1_MEMORY_PROOF_SCHEMA_VERSION",
    "OnlineO1Config",
    "OnlineO1Snapshot",
    "OnlineO1Synapse",
    "RewardModulatedLearner",
    "ShortTermPlasticity",
    "SNNGeneticEvolver",
    "StepScheduler",
    "StructuralPlasticity",
    "TensorBoardCallback",
    "TrainingCallback",
    "WandBCallback",
    "WarmupCosineScheduler",
    "build_online_o1_memory_proof",
]
