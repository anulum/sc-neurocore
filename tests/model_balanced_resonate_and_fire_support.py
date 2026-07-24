# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_balanced_resonate_and_fire.py

from __future__ import annotations

"""Publication-equation tests for Higuchi et al. 2024 BRF neuron."""
import math
from pathlib import Path
import numpy as np
import pytest
from sc_neurocore.neurons import BalancedResonateAndFireNeuron as PublicBRF
from sc_neurocore.neurons.models.balanced_resonate_and_fire import (
    BalancedResonateAndFireNeuron,
    sustain_oscillation_boundary,
)
from sc_neurocore.network.population import Population

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(neuron: BalancedResonateAndFireNeuron, current: float, steps: int) -> list[int]:
    return [step for step in range(steps) if neuron.step(current) == 1]


__all__ = [
    "math",
    "Path",
    "np",
    "pytest",
    "PublicBRF",
    "BalancedResonateAndFireNeuron",
    "sustain_oscillation_boundary",
    "Population",
    "REPO_ROOT",
    "_run",
]
