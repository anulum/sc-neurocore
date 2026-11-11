# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Spike-train power spectrum, re-exported from the shipped core

"""Spike-train power spectrum, re-exported from :mod:`sc_neurocore.spike_train_statistics`.

The implementation lives in the shipped package so that shipped modules can use
it from a wheel; this source-checkout path keeps the same name.
"""

from __future__ import annotations

from sc_neurocore.spike_train_statistics.spectral import power_spectrum

__all__ = ["power_spectrum"]
