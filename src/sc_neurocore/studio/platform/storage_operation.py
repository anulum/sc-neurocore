# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — bounded storage operation classification

"""Classify one already bounded metadata frame without accepting ambiguity."""

from __future__ import annotations

import json
from typing import Literal

StorageOperation = Literal[
    "record", "admit_named", "supervision", "finish", "query", "cancel", "artifact", "purge"
]


def _unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate names at every JSON object depth."""
    fields: dict[str, object] = {}
    for name, value in pairs:
        if name in fields:
            raise ValueError("duplicate storage operation field")
        fields[name] = value
    return fields


def _reject_constant(value: str) -> None:
    """Reject nonfinite JSON extensions before route selection."""
    raise ValueError("nonfinite storage operation constant")


def classify_storage_operation(metadata: bytes) -> StorageOperation:
    """Select one exact versioned operation from a peer-verified frame.

    The owning listener limits and authenticates the frame before calling this
    function. Each selected handler still validates the full operation schema.
    """
    try:
        parsed = json.loads(
            metadata.decode("utf-8"),
            object_pairs_hook=_unique_fields,
            parse_constant=_reject_constant,
        )
    except (UnicodeError, RecursionError, json.JSONDecodeError) as exc:
        raise ValueError("invalid storage operation JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("storage operation must be an object")
    version, operation = parsed.get("schema_version"), parsed.get("operation")
    if version == "studio.storage.record.v2" and operation == "record":
        return "record"
    if version == "studio.storage.admission.v1" and operation == "admit_named":
        return "admit_named"
    if version == "studio.storage.supervision.v1" and operation in ("start", "heartbeat"):
        return "supervision"
    if version == "studio.storage.finish.v1" and operation == "finish":
        return "finish"
    if version == "studio.storage.query.v1" and operation == "query":
        return "query"
    if version == "studio.storage.cancel.v1" and operation == "cancel":
        return "cancel"
    if version == "studio.storage.artifact.v1" and operation == "artifact":
        return "artifact"
    if version == "studio.storage.purge.v1" and operation == "purge":
        return "purge"
    raise ValueError("unsupported storage operation")
