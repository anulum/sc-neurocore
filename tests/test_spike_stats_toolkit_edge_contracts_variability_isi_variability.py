# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (isi_variability) from former test_spike_stats_toolkit_edge_contracts_variability.py

from __future__ import annotations

from tests.spike_stats_toolkit_edge_contracts_support import *  # noqa: F403

def test_cv_isi_zero():
    assert np.isnan(cv_isi(np.zeros(10, dtype=np.int8)))


def test_cv2_zero():
    assert np.isnan(cv2(np.zeros(10, dtype=np.int8)))


def test_lv_zero():
    assert np.isnan(local_variation(np.zeros(10, dtype=np.int8)))


def test_lvr_zero():
    assert np.isnan(lvr(np.zeros(10, dtype=np.int8)))


def test_fano_short():
    assert np.isnan(fano_factor(np.zeros(2, dtype=np.int8), window_ms=100.0))
