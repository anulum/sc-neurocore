# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Compatibility namespace for legacy campaign imports

"""Compatibility namespace for bridge-facing campaign imports."""

from __future__ import annotations

from .bridge import (
    QPUBridgeArtifact,
    SourceDataUnavailable,
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
)

__all__ = [
    "QPUBridgeArtifact",
    "SourceDataUnavailable",
    "load_connectome",
    "load_live_stream",
    "load_power_grid",
    "load_tokamak_data",
]
