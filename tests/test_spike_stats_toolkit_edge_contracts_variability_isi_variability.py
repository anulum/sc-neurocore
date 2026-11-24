# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (isi_variability) from former test_spike_stats_toolkit_edge_contracts_variability.py

"""The ISI variability measures and the Fano factor on silent or short trains."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.variability import (
    cv2,
    cv_isi,
    fano_factor,
    local_variation,
    lvr,
)


def test_cv_isi_zero() -> None:
    """The ISI coefficient of variation of a silent train is NaN."""
    assert np.isnan(cv_isi(np.zeros(10, dtype=np.int8)))


def test_cv2_zero() -> None:
    """The CV2 of a silent train is NaN."""
    assert np.isnan(cv2(np.zeros(10, dtype=np.int8)))


def test_lv_zero() -> None:
    """The local variation of a silent train is NaN."""
    assert np.isnan(local_variation(np.zeros(10, dtype=np.int8)))


def test_lvr_zero() -> None:
    """The revised local variation of a silent train is NaN."""
    assert np.isnan(lvr(np.zeros(10, dtype=np.int8)))


def test_fano_short() -> None:
    """The Fano factor of a train shorter than its window is NaN."""
    assert np.isnan(fano_factor(np.zeros(2, dtype=np.int8), window_ms=100.0))
