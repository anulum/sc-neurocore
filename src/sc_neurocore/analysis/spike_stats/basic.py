# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Basic spike-train operations, re-exported from the shipped core

"""Basic spike-train operations, re-exported from :mod:`sc_neurocore.spike_train_statistics`.

The implementations live in the shipped package so that shipped modules can use
them from a wheel; this source-checkout path keeps the same names.
"""

from __future__ import annotations

from sc_neurocore.spike_train_statistics.basic import (
    bin_spike_train,
    firing_rate,
    isi,
    spike_count,
    spike_times,
)

__all__ = ["bin_spike_train", "firing_rate", "isi", "spike_count", "spike_times"]
