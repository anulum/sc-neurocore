# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_contrastive.py

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import pytest
from sc_neurocore.contrastive import CSDPRule, SpikeContrastiveLoss

FloatArray = NDArray[np.float64]

__all__ = ["np", "NDArray", "pytest", "CSDPRule", "SpikeContrastiveLoss", "FloatArray"]
