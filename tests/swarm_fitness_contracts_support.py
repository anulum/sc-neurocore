# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_swarm_fitness_contracts.py

from __future__ import annotations

"""Exercise swarm.swarm_env, swarm.collective_fields, and swarm.fitness contracts."""
import numpy as np
import pytest
from sc_neurocore.swarm.swarm_env import SwarmEnvironment, EnvConfig
from sc_neurocore.swarm.collective_fields import (
    CollectiveFields,
    FieldConfig,
    _apply_laplacian,
)
from sc_neurocore.swarm.fitness import SwarmFitness

__all__ = ['np', 'pytest', 'SwarmEnvironment', 'EnvConfig', 'CollectiveFields', 'FieldConfig', '_apply_laplacian', 'SwarmFitness']
