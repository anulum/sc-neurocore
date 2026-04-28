# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Optimiser observation evidence loader

"""Load measured optimiser observations from benchmark and synthesis JSON.

The loader is intentionally strict.  It only creates
``BenchmarkObservation`` objects when the input carries both compiler design
settings and measured resource/performance values.  Missing fields raise an
error rather than being filled with invented data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sc_neurocore.optimizer.surrogate_sc_optimizer import BenchmarkObservation


class ObservationLoadError(ValueError):
    """Raised when a benchmark/synthesis observation cannot be trusted."""


def load_observations(path: str | Path) -> list[BenchmarkObservation]:
    """Load benchmark observations from a JSON evidence file."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ObservationLoadError(f"{source} is not valid JSON: {exc}") from exc
    return observations_from_payload(payload, source=str(source))


def observations_from_payload(
    payload: Any, *, source: str = "<memory>"
) -> list[BenchmarkObservation]:
    """Convert an in-memory benchmark/synthesis payload into observations."""
    defaults: dict[str, Any] = {}
    if isinstance(payload, dict):
        defaults = _mapping(payload.get("design_defaults") or payload.get("design") or {})

    records = _extract_records(payload)
    if not records:
        raise ObservationLoadError(f"{source} contains no observation records")

    observations = []
    for index, record in enumerate(records):
        merged = dict(defaults)
        merged.update(_mapping(record))
        observations.append(_observation_from_record(merged, source=source, index=index))
    return observations


def _extract_records(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [_mapping(item) for item in payload]
    if not isinstance(payload, dict):
        raise ObservationLoadError("observation payload must be a JSON object or list")

    for key in ("observations", "benchmark_observations", "layers", "runs", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [_mapping(item) for item in value]

    if "report" in payload or "measurement" in payload or "luts_used" in payload:
        return [payload]
    return []


def _observation_from_record(
    record: dict[str, Any], *, source: str, index: int
) -> BenchmarkObservation:
    design = _merged_views(record, "candidate", "compiler", "layer", "design")
    measurement = _merged_views(record, "measurement", "report", "resources", "timing", "power")
    view = dict(record)
    view.update(design)
    view.update(measurement)

    return BenchmarkObservation(
        mac_count=_required_int(view, "mac_count", source, index),
        bitstream_length=_required_int(view, "bitstream_length", source, index),
        decorrelator=_required_str(view, "decorrelator", source, index),
        mode=_required_str(view, "mode", source, index),
        precision_bits=_required_int(view, "precision_bits", source, index),
        lfsr_polynomial=_required_str(view, "lfsr_polynomial", source, index),
        luts_used=_required_int_any(
            view,
            ("luts_used", "luts", "lut", "clb_luts", "logic_luts", "alm"),
            source,
            index,
        ),
        power_mw=_required_float_any(
            view,
            ("power_mw", "total_power_mw", "total_on_chip_power_mw", "thermal_power_mw"),
            source,
            index,
        ),
        latency_cycles=_required_int_any(
            view,
            ("latency_cycles", "cycles", "latency"),
            source,
            index,
        ),
        accuracy_score=_required_float_any(
            view,
            ("accuracy_score", "accuracy", "score", "parity_score"),
            source,
            index,
        ),
        is_critical_path=bool(view.get("is_critical_path", False)),
    )


def _merged_views(record: dict[str, Any], *keys: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in keys:
        value = record.get(key)
        if isinstance(value, dict):
            merged.update(value)
    return merged


def _mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ObservationLoadError("observation record must be a JSON object")
    return value


def _required_int(view: dict[str, Any], key: str, source: str, index: int) -> int:
    return _to_int(_required(view, key, source, index), key, source, index)


def _required_int_any(view: dict[str, Any], keys: tuple[str, ...], source: str, index: int) -> int:
    key, value = _required_any(view, keys, source, index)
    return _to_int(value, key, source, index)


def _required_float_any(
    view: dict[str, Any], keys: tuple[str, ...], source: str, index: int
) -> float:
    key, value = _required_any(view, keys, source, index)
    return _to_float(value, key, source, index)


def _required_str(view: dict[str, Any], key: str, source: str, index: int) -> str:
    value = _required(view, key, source, index)
    if not isinstance(value, str) or not value:
        raise ObservationLoadError(f"{source} observation {index}: {key} must be a string")
    return value


def _required(view: dict[str, Any], key: str, source: str, index: int) -> Any:
    if key not in view:
        raise ObservationLoadError(f"{source} observation {index}: missing {key}")
    return view[key]


def _required_any(
    view: dict[str, Any], keys: tuple[str, ...], source: str, index: int
) -> tuple[str, Any]:
    for key in keys:
        if key in view:
            return key, view[key]
    joined = ", ".join(keys)
    raise ObservationLoadError(f"{source} observation {index}: missing one of {joined}")


def _to_int(value: Any, key: str, source: str, index: int) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ObservationLoadError(f"{source} observation {index}: {key} must be an int") from exc
    if result < 0:
        raise ObservationLoadError(f"{source} observation {index}: {key} must be non-negative")
    return result


def _to_float(value: Any, key: str, source: str, index: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ObservationLoadError(f"{source} observation {index}: {key} must be numeric") from exc
    if result < 0.0:
        raise ObservationLoadError(f"{source} observation {index}: {key} must be non-negative")
    return result
