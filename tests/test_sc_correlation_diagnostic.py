# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-correlation diagnostic tests

"""Correlated-pair flagging and independent-pair acceptance contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import CorrelationDiagnostic, correlation_diagnostic
from sc_neurocore.encoding.encoders import rate_encode
from tests.sc_correlation_support import _shared_source_streams


def test_diagnostic_flags_correlated_pair() -> None:
    a, b = _shared_source_streams(0.6, 0.4, 5000, seed=8)
    result = correlation_diagnostic(a, b, bias_threshold=0.01)
    assert isinstance(result, CorrelationDiagnostic)
    assert result.flagged is True
    assert result.scc == pytest.approx(1.0, abs=1e-9)
    assert result.predicted_and_bias == pytest.approx(result.observed_and_bias, abs=1e-9)


def test_diagnostic_passes_independent_pair() -> None:
    a = rate_encode(np.full(1, 0.5), T=5000, seed=20)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=5000, seed=21)[:, 0].astype(np.uint8)
    result = correlation_diagnostic(a, b, bias_threshold=0.05)
    assert result.flagged is False
