# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_brunel_translator.py

from __future__ import annotations

"""Tests for Brian2 ↔ SC-NeuroCore Brunel parameter translator."""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "benchmarks"))
from brunel_translator import (
    BrunelParams,
    translate_v1_stochastic_lif,
    translate_v2_rate_matched,
    translate_v3_fixed_point,
    translate_v4_hybrid,
    translate_v5_izhikevich,
    translate_v6_homeostatic,
    translate_v7_noisy,
    translate_v8_refractory,
    translate_v9_post_kick,
    translate_v10_exact_leak,
    translate_v11_q16,
    translate_v12_stdp,
    translate_v13_dot_product,
    translate_v14_sobol,
    translate_v15_jax,
    translate_v16_recurrent,
    translate_v17_memristive,
    translate_v18_numba,
    translate_v19_pytorch_cuda,
    translate_v20_vectorized_numpy,
)
from sc_neurocore import (
    StochasticLIFNeuron,
    FixedPointLIFNeuron,
    BitstreamSynapse,
    VectorizedSCLayer,
    SCIzhikevichNeuron,
    HomeostaticLIFNeuron,
)

__all__ = ['os', 'sys', 'np', 'BrunelParams', 'translate_v1_stochastic_lif', 'translate_v2_rate_matched', 'translate_v3_fixed_point', 'translate_v4_hybrid', 'translate_v5_izhikevich', 'translate_v6_homeostatic', 'translate_v7_noisy', 'translate_v8_refractory', 'translate_v9_post_kick', 'translate_v10_exact_leak', 'translate_v11_q16', 'translate_v12_stdp', 'translate_v13_dot_product', 'translate_v14_sobol', 'translate_v15_jax', 'translate_v16_recurrent', 'translate_v17_memristive', 'translate_v18_numba', 'translate_v19_pytorch_cuda', 'translate_v20_vectorized_numpy', 'StochasticLIFNeuron', 'FixedPointLIFNeuron', 'BitstreamSynapse', 'VectorizedSCLayer', 'SCIzhikevichNeuron', 'HomeostaticLIFNeuron']
