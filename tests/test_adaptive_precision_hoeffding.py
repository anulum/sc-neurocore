# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Adaptive precision planner contracts

"""Focused adaptive precision planner contracts."""

from __future__ import annotations


import pytest

from sc_neurocore.compiler.synapse_planner import _hoeffding_radius


def test_hoeffding_radius_rejects_non_positive_length() -> None:
    """The internal Hoeffding helper fails closed for invalid lengths."""
    with pytest.raises(ValueError, match="length"):
        _hoeffding_radius(0, 0.95)


def test_hoeffding_radius_rejects_invalid_confidence() -> None:
    """The internal Hoeffding helper rejects invalid confidence values."""
    with pytest.raises(ValueError, match="confidence"):
        _hoeffding_radius(16, 1.0)
