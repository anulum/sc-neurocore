# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for hardware package contracts

"""Contracts for public hardware package metadata."""

from __future__ import annotations


def test_hardware_package_exposes_core_tier() -> None:
    import sc_neurocore.hardware

    assert sc_neurocore.hardware.__tier__ == "core"
