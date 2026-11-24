# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rate) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of the PSTH with empty trials."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.rate import (
    psth,
)


def test_psth_zero_bins() -> None:
    """The PSTH of one empty trial is empty."""
    r, t = psth([np.array([], dtype=np.int8)], bin_ms=100.0)
    assert r.size == 0


def test_psth_empty_trial() -> None:
    """The PSTH of a full and an empty trial still has bins."""
    r, t = psth([np.zeros(200, dtype=np.int8), np.array([], dtype=np.int8)], bin_ms=10.0)
    assert r.shape[0] > 0
