# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_autofit.py

from __future__ import annotations

import numpy as np
import pytest
from sc_neurocore.autofit.features import extract_spike_times, extract_features
from sc_neurocore.autofit.fitter import (
    FittedModel,
    _cost_rmse,
    _cost_features,
    _simulate,
    _fit_single_model,
    _get_model_class,
    fit,
)

__all__ = [
    "np",
    "pytest",
    "extract_spike_times",
    "extract_features",
    "FittedModel",
    "_cost_rmse",
    "_cost_features",
    "_simulate",
    "_fit_single_model",
    "_get_model_class",
    "fit",
]
