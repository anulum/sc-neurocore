# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Thermal SC encoding test support

"""Shared imports and baseline helper for thermal SC encoding tests."""

from __future__ import annotations

import pytest

from sc_neurocore.security.side_channel_metrics import compute_class_activity_proxy
from sc_neurocore.security.thermal_sc_encoding import (
    ThermalSCEncodingConfig,
    ThermalSCEncodingError,
    _activity_preserving_rotation_offset,
    _distribute_ones,
    encode_activity_balanced_probability,
    encode_activity_balanced_probabilities,
)

__all__ = [
    "ThermalSCEncodingConfig",
    "ThermalSCEncodingError",
    "_activity_preserving_rotation_offset",
    "_correlated_activity_stream",
    "_distribute_ones",
    "compute_class_activity_proxy",
    "encode_activity_balanced_probabilities",
    "encode_activity_balanced_probability",
    "pytest",
]


def _correlated_activity_stream(probability: float, bitstream_length: int) -> tuple[int, ...]:
    ones = round(probability * bitstream_length)
    if probability >= 0.5:
        return tuple(index % 2 for index in range(bitstream_length))
    return tuple(1 if index < ones else 0 for index in range(bitstream_length))
