# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Layer precision specification

"""Per-layer bitstream length assignment specification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LayerPrecision:
    """Bitstream length assignment for one layer."""

    layer_index: int
    name: str
    bitstream_length: int
    error_bound: float
    sensitivity: float
