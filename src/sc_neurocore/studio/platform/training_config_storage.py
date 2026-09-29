# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bounded training configuration and event input custody

"""Store large event contracts separately from the 4096-byte job configuration."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

CONFIG_MAX_BYTES = 4096
EVENT_DATA_MAX_BYTES = 64 * 1024 * 1024
REFERENCE_KEY = "event_data_reference"
REFERENCE_SCHEMA = "studio.event-input-reference.v1"


def _canonical(value: object) -> str:
    """Encode one JSON value with the ledger's deterministic ordering."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _reference(value: str) -> dict[str, object]:
    """Bind the complete UTF-8 event declaration by byte count and SHA-256."""
    raw = value.encode("utf-8")
    return {
        "schema": REFERENCE_SCHEMA,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def prepare_training_config(config: Mapping[str, object]) -> tuple[str, str | None]:
    """Validate a public configuration and separate its complete event contract.

    Parameters
    ----------
    config : mapping
        The complete public training configuration submitted with the payload.

    Returns
    -------
    tuple
        Canonical small configuration and an optional canonical event contract.
        Both belong to the same job admission transaction.

    Raises
    ------
    ValueError
        The training declaration is invalid or either independent byte limit
        is exceeded. Large hidden-layer declarations retain the 4096-byte limit.
    """
    from sc_neurocore.studio.training_contract import resolve_training_config

    resolved = dict(resolve_training_config(config).to_public_dict())
    event_data = resolved.get("event_data")
    event_json = None
    if event_data is not None:
        event_json = _canonical(event_data)
        if len(event_json.encode("utf-8")) > EVENT_DATA_MAX_BYTES:
            raise ValueError("Event input declaration exceeds the 64 MiB custody limit.")
        del resolved["event_data"]
        resolved[REFERENCE_KEY] = _reference(event_json)
    config_json = _canonical(resolved)
    if len(config_json.encode("utf-8")) > CONFIG_MAX_BYTES:
        raise ValueError("Training configuration exceeds the 4096-byte admission limit.")
    return config_json, event_json


def restore_training_config(payload: dict[str, Any], event_json: str | None) -> dict[str, Any]:
    """Verify stored event bytes before rebuilding a full public configuration.

    Parameters
    ----------
    payload : dict
        Decoded configuration row, either a legacy inline value or a reference.
    event_json : str, optional
        The same row's separate immutable event declaration.

    Returns
    -------
    dict
        Complete public configuration; no internal reference is exposed.

    Raises
    ------
    ValueError
        A declaration is absent, altered, oversized, malformed or noncanonical,
        or a legacy row contains unexpected separate event data.
    """
    if REFERENCE_KEY not in payload:
        if event_json is not None:
            raise ValueError("event input data has no configuration reference")
        return payload
    if event_json is None:
        raise ValueError("referenced event input data is absent")
    if len(event_json.encode("utf-8")) > EVENT_DATA_MAX_BYTES:
        raise ValueError("stored event input data exceeds the custody limit")
    if payload[REFERENCE_KEY] != _reference(event_json):
        raise ValueError("stored event input data does not match its reference")
    event_data = json.loads(event_json)
    if not isinstance(event_data, dict) or _canonical(event_data) != event_json:
        raise ValueError("stored event input data is not a canonical object")
    if "event_data" in payload:
        raise ValueError("stored configuration mixes inline and referenced event data")
    restored = dict(payload)
    del restored[REFERENCE_KEY]
    restored["event_data"] = event_data
    return restored
