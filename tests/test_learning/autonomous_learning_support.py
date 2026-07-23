# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_autonomous_learning.py

from __future__ import annotations

import numpy as np
import pytest
try:
    from sc_neurocore._native.learning_bridge import (
        is_available,
        RustPlasticityRule,
        RustEligentLearner,
        RustRuleLayer,
        RustOnlineO1Synapse,
        RULE_ELIGENT,
        RULE_STDP,
        RULE_REWARD_STDP,
        RULE_BCM,
        create_plasticity_layer,
    )

    FFI_AVAILABLE = is_available()
except ImportError:
    FFI_AVAILABLE = False
from sc_neurocore.meta_plasticity.meta_plasticity import MetaPlasticityEngine, EngineConfig

__all__ = ['FFI_AVAILABLE', 'np', 'pytest', 'is_available', 'RustPlasticityRule', 'RustEligentLearner', 'RustRuleLayer', 'RustOnlineO1Synapse', 'RULE_ELIGENT', 'RULE_STDP', 'RULE_REWARD_STDP', 'RULE_BCM', 'create_plasticity_layer', 'MetaPlasticityEngine', 'EngineConfig']
