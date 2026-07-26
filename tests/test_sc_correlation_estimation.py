# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Stochastic-correlation estimation tests

"""Independent, comonotone, countermonotone, and degenerate SCC contracts."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core import estimate_scc
from sc_neurocore.encoding.encoders import rate_encode
from tests.sc_correlation_support import _shared_source_streams


def test_scc_independent_streams_near_zero() -> None:
    a = rate_encode(np.full(1, 0.5), T=4000, seed=1)[:, 0].astype(np.uint8)
    b = rate_encode(np.full(1, 0.5), T=4000, seed=2)[:, 0].astype(np.uint8)
    assert abs(estimate_scc(a, b)) < 0.1


def test_scc_comonotone_is_plus_one() -> None:
    a, b = _shared_source_streams(0.6, 0.4, 5000, seed=3)
    assert estimate_scc(a, b) == pytest.approx(1.0, abs=1e-9)


def test_scc_countermonotone_is_minus_one() -> None:
    rng = np.random.default_rng(4)
    u = rng.random(5000)
    a = (u < 0.6).astype(np.uint8)
    b = (u >= 1.0 - 0.4).astype(np.uint8)  # inverted source -> anti-correlated
    assert estimate_scc(a, b) == pytest.approx(-1.0, abs=1e-9)


def test_scc_degenerate_stream_returns_zero() -> None:
    ones = np.ones(100, dtype=np.uint8)
    mixed = np.tile([1, 0], 50).astype(np.uint8)
    assert estimate_scc(ones, mixed) == 0.0
