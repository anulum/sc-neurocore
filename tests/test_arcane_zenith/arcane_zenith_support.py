# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_arcane_zenith.py

from __future__ import annotations

"""Multi-angle tests for ``sc_neurocore.arcane_zenith``.

The ArcaneZenith core glues an ``ArcaneNeuron`` (5-compartment
self-referential cognition neuron) to four reward-modulated STDP
plasticity rules whose scalar weights ``w ∈ [0, 1]`` are mapped via a
sharpened sigmoid to biologically plausible ranges for the neuron's
four meta-parameters:

    tau_deep           ∈ [1000, 50000] ms
    surprise_baseline  ∈ [0.01, 0.5]
    delta_conf         ∈ [0, 1]
    lr_base            ∈ [0.001, 0.1]

Tests cover: the sigmoid mapping itself (monotonicity, endpoints,
midpoint), construction via factory + direct, the ``step`` contract,
biological-range invariants across many steps, ``step_from_bio_rates``
with arbitrary firing-rate dicts (including empty), ``reset`` semantics
(spike compartments clear, identity persists), ``get_state`` /
``get_state_dict`` round-trip, and an end-to-end stability check.

Tests use the ``"torch"`` plasticity backend so the suite runs on
machines without ``libautonomous_learning``.
"""
import math
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from sc_neurocore.arcane_zenith import (
    ArcaneZenithCognitiveCore,
    create_arcane_neuron_with_zenith_plasticity,
)
from sc_neurocore.fault_injection import RadiationProfile
from sc_neurocore.neurons.models.arcane_neuron import ArcaneNeuron

__all__ = [
    "math",
    "np",
    "pytest",
    "torch",
    "ArcaneZenithCognitiveCore",
    "create_arcane_neuron_with_zenith_plasticity",
    "RadiationProfile",
    "ArcaneNeuron",
    "__all__",
]
