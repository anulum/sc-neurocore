# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Shipped spike-train statistics core

"""Shipped spike-train statistics: the NumPy-only core of the spike statistics.

This package ships in the wheel, so shipped modules such as
:mod:`sc_neurocore.identity` can measure spike trains in an installed copy.
:mod:`sc_neurocore.analysis.spike_stats`, which ships only in a source
checkout, re-exports these functions under its own names and adds the wider
toolkit. This package imports nothing from :mod:`sc_neurocore.analysis`.
"""

from __future__ import annotations

from .basic import bin_spike_train, firing_rate, isi, spike_count, spike_times
from .spectral import power_spectrum
from .variability import cv_isi, fano_factor, permutation_entropy

__all__ = [
    "bin_spike_train",
    "cv_isi",
    "fano_factor",
    "firing_rate",
    "isi",
    "permutation_entropy",
    "power_spectrum",
    "spike_count",
    "spike_times",
]
