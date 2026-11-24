# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (surrogates) from former test_spike_stats_toolkit_edge_contracts.py

"""Edge contracts of the surrogate spike-train generators and of response-onset detection."""

from __future__ import annotations

import numpy as np
from sc_neurocore.analysis.spike_stats.surrogates import (
    gamma_process,
    homogeneous_poisson,
    surrogate_isi_shuffle,
    surrogate_joint_isi,
)
from sc_neurocore.analysis.spike_stats.temporal import (
    response_onset,
)


def test_isi_shuffle_short() -> None:
    """The ISI shuffle of a one-sample train keeps its length."""
    r = surrogate_isi_shuffle(np.array([1], dtype=np.int8))
    assert r.shape[0] == 1


def test_poisson_zero_rate() -> None:
    """A homogeneous Poisson train at zero rate is silent."""
    r = homogeneous_poisson(rate_hz=0.0, duration_s=1.0)
    assert np.all(r == 0)


def test_gamma_zero_rate() -> None:
    """A gamma-process train at zero rate is silent."""
    r = gamma_process(rate_hz=0.0, shape=2, duration_s=1.0)
    assert np.all(r == 0)


def test_joint_isi_few() -> None:
    """The joint-ISI surrogate of a single spike keeps the train length."""
    r = surrogate_joint_isi(np.array([1, 0, 0], dtype=np.int8))
    assert r.shape[0] == 3


def test_response_onset_short() -> None:
    """The response onset of a train shorter than its baseline is NaN."""
    r = response_onset(np.array([1, 0], dtype=np.int8), baseline_steps=5)
    assert np.isnan(r)
