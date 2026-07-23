# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_sc_optimizer_extended.py

from __future__ import annotations

import unittest
from sc_neurocore.optimizer.sc_optimizer import (
    SCOptimizer,
    HardwareBudget,
    LayerProfile,
    OptimizerReport,
    DecorrelationStrategy,
    ComputeMode,
)
def make_network(n: int = 5, mac: int = 100) -> list[LayerProfile]:
    return [LayerProfile(id=f"L{i}", mac_count=mac, is_critical_path=(i == 0)) for i in range(n)]

__all__ = ['unittest', 'SCOptimizer', 'HardwareBudget', 'LayerProfile', 'OptimizerReport', 'DecorrelationStrategy', 'ComputeMode', 'make_network']
