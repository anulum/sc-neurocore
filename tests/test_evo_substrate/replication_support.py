# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_replication.py

from __future__ import annotations

"""Evolutionary replication workflow tests."""
from typing import Any, cast
import numpy as np
import pytest
import sc_neurocore.evo_substrate.evo_substrate as evo_mod
from sc_neurocore.evo_substrate.ecology import ExtinctionDetector
from sc_neurocore.evo_substrate.fitness import FitnessResult
from sc_neurocore.evo_substrate.genome import Genome
from sc_neurocore.evo_substrate.organism import Organism
from sc_neurocore.evo_substrate.replication import ReplicationEngine
from sc_neurocore.evo_substrate.safety import FormalSafetyGuard, SafetyBounds
from sc_neurocore.fault_injection import DegradationPlan, FaultModel
from sc_neurocore.fault_injection.resilience_policy import SeededFaultObservation
from sc_neurocore.stochastic_doctor.diagnostics import AuditSeverity, BitstreamAuditReport

__all__ = ['Any', 'cast', 'np', 'pytest', 'evo_mod', 'ExtinctionDetector', 'FitnessResult', 'Genome', 'Organism', 'ReplicationEngine', 'FormalSafetyGuard', 'SafetyBounds', 'DegradationPlan', 'FaultModel', 'SeededFaultObservation', 'AuditSeverity', 'BitstreamAuditReport']
