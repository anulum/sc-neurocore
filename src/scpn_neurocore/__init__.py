# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Canonical SCPN bridge namespace

"""Canonical namespace for bridge-facing SCPN campaign imports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from .bridge import (
    QPUBridgeArtifact,
    SourceDataUnavailable,
    load_connectome,
    load_live_stream,
    load_power_grid,
    load_tokamak_data,
    validate_qpu_artifact_payload,
)

_DATASTREAM_EXPORTS = frozenset(
    {
        "SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION",
        "DatastreamValidationError",
        "SCNeuroCoreDatastreamPacket",
        "build_datastream_packet",
        "validate_datastream_payload",
    }
)


def __getattr__(name: str) -> Any:
    if name in _DATASTREAM_EXPORTS:
        datastream = import_module(".datastream", __name__)
        value = getattr(datastream, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if TYPE_CHECKING:
    from .datastream import (
        SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION,
        DatastreamValidationError,
        SCNeuroCoreDatastreamPacket,
        build_datastream_packet,
        validate_datastream_payload,
    )

__all__ = [
    "SC_NEUROCORE_DATASTREAM_SCHEMA_VERSION",
    "DatastreamValidationError",
    "SCNeuroCoreDatastreamPacket",
    "QPUBridgeArtifact",
    "SourceDataUnavailable",
    "build_datastream_packet",
    "load_connectome",
    "load_live_stream",
    "load_power_grid",
    "load_tokamak_data",
    "validate_datastream_payload",
    "validate_qpu_artifact_payload",
]
