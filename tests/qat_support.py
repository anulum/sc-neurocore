# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_qat.py

from __future__ import annotations

import numpy as np
from sc_neurocore.qat import QuantizedSNNLayer, quantize_aware_train_step, TernaryWeights
from sc_neurocore.qat.quantize import _ste_quantize

__all__ = ['np', 'QuantizedSNNLayer', 'quantize_aware_train_step', 'TernaryWeights', '_ste_quantize']
