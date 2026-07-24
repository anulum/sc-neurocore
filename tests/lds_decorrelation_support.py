# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_lds_decorrelation.py

from __future__ import annotations

"""Tests for multi-dimensional Sobol/Halton decorrelation."""
import numpy as np
import pytest
from sc_neurocore.utils.lds_decorrelation import (
    generate_decorrelated_bitstreams,
    star_discrepancy_estimate,
)
import scipy.stats.qmc as qmc

__all__ = ["np", "pytest", "generate_decorrelated_bitstreams", "star_discrepancy_estimate", "qmc"]
