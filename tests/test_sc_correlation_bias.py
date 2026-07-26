# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-correlation bias tests

"""Observed AND bias and WC-C3 prediction consistency contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import estimate_scc, observed_and_bias
from sc_neurocore.core.sc_error_bounds import multiply_correlation_bias
from sc_neurocore.encoding.encoders import rate_encode
from tests.sc_correlation_support import _shared_source_streams


def test_observed_bias_zero_for_independent() -> None:
    a = rate_encode(np.full(1, 0.6), T=4000, seed=10)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=4000, seed=11)[:, 0].astype(np.uint8)
    assert abs(observed_and_bias(a, b)) < 0.02


def test_scc_predicts_observed_and_bias_comonotone() -> None:
    # The SCC measured from the streams must reproduce the observed AND bias
    # through the WC-C3 multiply_correlation_bias formula.
    p_a, p_b = 0.6, 0.4
    a, b = _shared_source_streams(p_a, p_b, 5000, seed=7)
    scc = estimate_scc(a, b)
    predicted = multiply_correlation_bias(float(np.mean(a)), float(np.mean(b)), scc)
    assert predicted == pytest.approx(observed_and_bias(a, b), abs=1e-9)
    # Comonotone bias equals min(p_a, p_b) - p_a * p_b.
    assert observed_and_bias(a, b) == pytest.approx(min(p_a, p_b) - p_a * p_b, abs=0.02)
