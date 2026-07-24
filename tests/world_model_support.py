# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_world_model.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.world_model.spike_predictor import (
    SpikePredictor,
    predict_and_xor_world_model,
    xor_and_recover_world_model,
)
import sc_neurocore.world_model as world_model_module

__all__ = [
    "np",
    "pytest",
    "SpikePredictor",
    "predict_and_xor_world_model",
    "xor_and_recover_world_model",
    "world_model_module",
]
