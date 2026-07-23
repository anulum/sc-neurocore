# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_meta_plasticity.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.meta_plasticity.meta_plasticity import (
    CheckpointStore,
    ContextRuleBank,
    CuriositySignal,
    EWCProtection,
    EngineConfig,
    FitnessTrajectory,
    HomeostaticParams,
    MetaControlSignal,
    MetaController,
    MetaLearningRate,
    MetaPlasticityEngine,
    MetaSignalType,
    NeuromodulatorState,
    NeuromodulatorType,
    PlasticityRuleSet,
    RuleConstraints,
    RuleEvolver,
    STDPParams,
    STPParams,
    SleepPhase,
    TaggingModel,
    inject_diversity,
    population_diversity,
)

__all__ = ['np', 'pytest', 'CheckpointStore', 'ContextRuleBank', 'CuriositySignal', 'EWCProtection', 'EngineConfig', 'FitnessTrajectory', 'HomeostaticParams', 'MetaControlSignal', 'MetaController', 'MetaLearningRate', 'MetaPlasticityEngine', 'MetaSignalType', 'NeuromodulatorState', 'NeuromodulatorType', 'PlasticityRuleSet', 'RuleConstraints', 'RuleEvolver', 'STDPParams', 'STPParams', 'SleepPhase', 'TaggingModel', 'inject_diversity', 'population_diversity']
