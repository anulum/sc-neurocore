# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (point_process) from former test_spike_stats_toolkit_edge_contracts.py

"""The ISI hazard, survivor and renewal-density functions of a train with a single spike."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.point_process import (
    isi_hazard_function,
    isi_survivor_function,
    renewal_density,
)


def test_hazard_few() -> None:
    """The ISI hazard function of a single spike is empty."""
    h, e = isi_hazard_function(np.array([1, 0], dtype=np.int8))
    assert h.size == 0


def test_survivor_few() -> None:
    """The ISI survivor function of a single spike is empty."""
    s, e = isi_survivor_function(np.array([1, 0], dtype=np.int8))
    assert s.size == 0


def test_renewal_few() -> None:
    """The renewal density of a single spike is empty."""
    r, e = renewal_density(np.array([1, 0], dtype=np.int8))
    assert r.size == 0
