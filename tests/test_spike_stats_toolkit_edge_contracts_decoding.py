# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (decoding) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contract of the Bayesian decoder with a single class."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.decoding import (
    bayesian_decode,
)


def test_bayesian_single_class() -> None:
    """Bayesian decoding with a single class returns class 0."""
    assert bayesian_decode(np.array([3.0]), np.array([[1.0]])) == 0
