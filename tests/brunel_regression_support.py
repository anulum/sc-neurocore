# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_brunel_regression.py

from __future__ import annotations

"""Regression and consistency tests derived from the 20-variant Brunel benchmark results.

These tests validate invariants that hold across variants without re-running
the full 1000ms simulation. They use the saved JSON artifact and lightweight
translator/neuron smoke runs.
"""
import json
import os
import sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v3_fixed_point,
    translate_v7_noisy,
    translate_v8_refractory,
    translate_v9_post_kick,
    translate_v10_exact_leak,
    translate_v11_q16,
    translate_v20_vectorized_numpy,
)
from sc_neurocore import StochasticLIFNeuron, FixedPointLIFNeuron

RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "benchmarks", "results", "snn_translator_20v.json"
)


def _load_results() -> dict[str, dict]:
    if not os.path.exists(RESULTS_PATH):
        pytest.skip("benchmark results JSON not found")
    with open(RESULTS_PATH) as f:
        data = json.load(f)
    return {r["variant"]: r for r in data}


__all__ = [
    "json",
    "os",
    "sys",
    "np",
    "pytest",
    "BrunelParams",
    "translate_v1_stochastic_lif",
    "translate_v3_fixed_point",
    "translate_v7_noisy",
    "translate_v8_refractory",
    "translate_v9_post_kick",
    "translate_v10_exact_leak",
    "translate_v11_q16",
    "translate_v20_vectorized_numpy",
    "StochasticLIFNeuron",
    "FixedPointLIFNeuron",
    "RESULTS_PATH",
    "_load_results",
]
