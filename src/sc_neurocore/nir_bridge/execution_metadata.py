# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Execution settings carried by the NIR metadata extension

"""Preserve the timestep and reset convention that NIR primitives do not encode.

The extension is JSON text inside ordinary NIR metadata, so the reference NIR
reader and writer carry it without changes to their graph or file schema.
Foreign graphs without this extension retain the historical default settings.
An explicit importer option cannot silently contradict a recorded setting.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass

EXECUTION_METADATA_KEY = "sc_neurocore.execution"
EXECUTION_METADATA_SCHEMA = "sc-neurocore.nir-execution.v1"


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    """Describe one discrete-time network execution convention.

    Parameters
    ----------
    dt : float
        Positive finite timestep in the same time unit as the NIR parameters.
    reset_mode : str
        Immediate reset to the declared reset potential, or threshold subtraction.
    """

    dt: float
    reset_mode: str


def _settings(dt: object, reset_mode: object) -> ExecutionSettings:
    if isinstance(dt, bool) or not isinstance(dt, (int, float)):
        raise ValueError("NIR import dt must be a finite number")
    if not math.isfinite(float(dt)):
        raise ValueError("NIR import dt must be a finite number")
    if dt <= 0:
        raise ValueError("NIR import dt must be positive")
    if not isinstance(reset_mode, str) or reset_mode not in (
        "reset",
        "subtract",
        "snntorch-subtract",
    ):
        raise ValueError(
            "NIR import reset_mode must be one of ('reset', 'subtract', 'snntorch-subtract')"
        )
    return ExecutionSettings(float(dt), reset_mode)


def _unique_members(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate NIR metadata member {key!r}")
        result[key] = value
    return result


def _recorded_settings(metadata: Mapping[str, object]) -> ExecutionSettings | None:
    if not isinstance(metadata, Mapping):
        raise ValueError("NIR execution metadata must be a mapping")
    if EXECUTION_METADATA_KEY not in metadata:
        return None
    text = metadata[EXECUTION_METADATA_KEY]
    if not isinstance(text, str):
        raise ValueError("NIR execution metadata must be JSON text")
    try:
        document: object = json.loads(text, object_pairs_hook=_unique_members)
    except (ValueError, RecursionError) as exc:
        raise ValueError("NIR execution metadata is not readable JSON") from exc
    if not isinstance(document, dict):
        raise ValueError("NIR execution metadata must describe an object")
    if document.get("schema_version") != EXECUTION_METADATA_SCHEMA:
        raise ValueError("NIR execution metadata uses an unsupported schema")
    if set(document) != {"schema_version", "dt", "reset_mode"}:
        raise ValueError("NIR execution metadata has missing or unknown fields")
    return _settings(document["dt"], document["reset_mode"])


def read_execution_settings(
    metadata: Mapping[str, object],
    *,
    dt: float | None = None,
    reset_mode: str | None = None,
) -> ExecutionSettings:
    """Resolve import options against the execution contract in NIR metadata.

    Parameters
    ----------
    metadata : mapping
        Graph metadata read by the reference NIR library.
    dt : float or None
        Requested timestep. A recorded timestep is used when this is absent.
    reset_mode : str or None
        Requested reset convention. A recorded convention is used when absent.

    Returns
    -------
    ExecutionSettings
        Recorded settings, explicit options, or the foreign-graph defaults of
        timestep one and immediate reset to the reset potential.

    Raises
    ------
    ValueError
        The recorded contract is malformed, options are invalid, or an option
        contradicts a setting already recorded in the graph.
    """
    recorded = _recorded_settings(metadata)
    selected = _settings(
        dt if dt is not None else recorded.dt if recorded is not None else 1.0,
        reset_mode
        if reset_mode is not None
        else recorded.reset_mode
        if recorded is not None
        else "reset",
    )
    if recorded is not None and selected != recorded:
        raise ValueError("NIR import options conflict with recorded execution settings")
    return selected


def write_execution_metadata(
    metadata: Mapping[str, object], *, dt: float, reset_mode: str
) -> dict[str, object]:
    """Copy graph metadata and record the current execution settings.

    Parameters
    ----------
    metadata : mapping
        Original metadata, retained independently of the source object.
    dt : float
        Positive finite timestep in the graph's declared time unit.
    reset_mode : str
        Immediate reset to the reset potential, or threshold subtraction.

    Returns
    -------
    dict
        Independent metadata with the versioned execution extension.

    Raises
    ------
    ValueError
        Existing execution metadata is not a recognised contract, or the
        supplied settings are invalid.
    """
    _recorded_settings(metadata)
    settings = _settings(dt, reset_mode)
    result = deepcopy(dict(metadata))
    result[EXECUTION_METADATA_KEY] = json.dumps(
        {
            "schema_version": EXECUTION_METADATA_SCHEMA,
            "dt": settings.dt,
            "reset_mode": settings.reset_mode,
        },
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return result
