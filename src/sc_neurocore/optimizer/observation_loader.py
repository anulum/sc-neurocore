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
import re
from pathlib import Path
from typing import Any, Mapping

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


def load_synthesis_observation(
    report_paths: Mapping[str, str | Path],
    *,
    design: Mapping[str, Any],
    accuracy_score: float,
    latency_cycles: int | None = None,
) -> BenchmarkObservation:
    """Load one observation from Vivado/Quartus report files plus design metadata.

    Raw vendor reports do not describe the compiler decision that produced the
    hardware, and many do not carry model accuracy.  The caller must therefore
    provide the design fields and measured accuracy explicitly.
    """
    reports: dict[str, str] = {}
    for name, path in report_paths.items():
        source = Path(path)
        reports[name] = source.read_text(encoding="utf-8", errors="ignore")
    return observation_from_synthesis_reports(
        reports,
        design=design,
        accuracy_score=accuracy_score,
        latency_cycles=latency_cycles,
        source=", ".join(str(path) for path in report_paths.values()),
    )


def observation_from_synthesis_reports(
    reports: Mapping[str, str],
    *,
    design: Mapping[str, Any],
    accuracy_score: float,
    latency_cycles: int | None = None,
    source: str = "<synthesis-reports>",
) -> BenchmarkObservation:
    """Build one observation from raw Vivado/Quartus text reports."""
    metrics = _metrics_from_synthesis_reports(reports, source=source)
    if latency_cycles is not None:
        metrics["latency_cycles"] = latency_cycles
    record = dict(design)
    record.update(metrics)
    record["accuracy_score"] = accuracy_score
    return _observation_from_record(record, source=source, index=0)


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


def _metrics_from_synthesis_reports(
    reports: Mapping[str, str], *, source: str
) -> dict[str, int | float]:
    merged = "\n".join(reports.values())
    metrics: dict[str, int | float] = {}

    luts = _first_numeric_match(
        merged,
        (
            r"\bCLB\s+LUTs\b[^\n\r\d]*(?P<value>[\d,]+)",
            r"\bSlice\s+LUTs\b[^\n\r\d]*(?P<value>[\d,]+)",
            r"\bLogic\s+LUTs\b[^\n\r\d]*(?P<value>[\d,]+)",
            r"\bALMs?\s+(?:needed|required|used)\b[^\n\r\d]*(?P<value>[\d,]+)",
            r"\bTotal\s+combinational\s+functions\b[^\n\r\d]*(?P<value>[\d,]+)",
        ),
    )
    if luts is not None:
        metrics["luts_used"] = int(luts)

    power_mw = _first_power_mw(merged)
    if power_mw is not None:
        metrics["power_mw"] = power_mw

    latency = _first_numeric_match(
        merged,
        (
            r"\bLatency\s*\(?cycles\)?\b[^\n\r\d]*(?P<value>[\d,]+)",
            r"\bLatency\s*:\s*(?P<value>[\d,]+)\s*cycles\b",
            r"\bcycles\b[^\n\r\d]*(?P<value>[\d,]+)",
        ),
    )
    if latency is not None:
        metrics["latency_cycles"] = int(latency)

    missing = [key for key in ("luts_used", "power_mw") if key not in metrics]
    if missing:
        joined = ", ".join(missing)
        raise ObservationLoadError(f"{source}: synthesis reports missing {joined}")
    return metrics


def _first_numeric_match(text: str, patterns: tuple[str, ...]) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return float(match.group("value").replace(",", ""))
    return None


def _first_power_mw(text: str) -> float | None:
    patterns = (
        r"\bTotal\s+On-Chip\s+Power\s*\((?P<label_unit>m?W)\)\s*[:|]?\s*(?P<value>[\d.]+)",
        r"\bTotal\s+On-Chip\s+Power\b[^\n\r\d]*(?P<value>[\d.]+)\s*(?P<unit>m?W)\b",
        r"\bTotal\s+thermal\s+power\s+dissipation\b[^\n\r\d]*(?P<value>[\d.]+)\s*(?P<unit>m?W)\b",
        r"\bThermal\s+Power\b[^\n\r\d]*(?P<value>[\d.]+)\s*(?P<unit>m?W)\b",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = float(match.group("value"))
            unit = (match.groupdict().get("unit") or match.groupdict()["label_unit"]).lower()
            return value if unit == "mw" else value * 1000.0
    return None


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
