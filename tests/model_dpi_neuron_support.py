# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_model_dpi_neuron.py

from __future__ import annotations

"""Fidelity and safety tests for the coupled current-mode DPI recurrence."""

import math


from typing import Any, cast


import numpy as np


import pytest


from sc_neurocore.neurons.models.dpi_neuron import DPINeuron


def _configured() -> DPINeuron:
    """Return a stable non-default contract exercising every maintained field."""
    return DPINeuron(
        i_mem=0.37,
        i_ahp=0.08,
        refractory_time=0.0,
        i_threshold=1.3,
        i_reset=0.2,
        i_rest=0.15,
        i_tau=0.9,
        i_g=1.4,
        i_tau_ahp=0.12,
        i_ga=0.8,
        i_spike=4.2,
        i_0=0.02,
        kappa=0.65,
        alpha=8.0,
        tau=7.0,
        tau_ahp=45.0,
        refractory_period=0.6,
        dt=0.05,
    )


def _events(neuron: DPINeuron, current: float, steps: int) -> list[int]:
    """Return zero-based spike indices from one direct reference-model run."""
    return [index for index in range(steps) if neuron.step(current) == 1]


__all__ = ["math", "Any", "cast", "np", "pytest", "DPINeuron", "_configured", "_events"]
