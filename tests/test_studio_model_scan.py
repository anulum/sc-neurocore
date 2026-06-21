# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio model scan hardening and cache behaviour

from __future__ import annotations

from typing import cast

import pytest

from sc_neurocore.studio import model_scan
from sc_neurocore.studio.model_scan import JsonValue


def _object_field(payload: dict[str, JsonValue], field_name: str) -> dict[str, JsonValue]:
    """Return a JSON object field from a model-scan response."""

    value = payload[field_name]
    assert isinstance(value, dict)
    return value


def _list_field(payload: dict[str, JsonValue], field_name: str) -> list[JsonValue]:
    """Return a JSON list field from a model-scan response."""

    value = payload[field_name]
    assert isinstance(value, list)
    return value


def test_scan_all_models_caches_successful_results(monkeypatch: pytest.MonkeyPatch) -> None:
    model_scan._CACHE.clear()
    calls: dict[str, int] = {"simulate": 0}

    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [{"name": "ModelA", "category": "CatA"}],
    )

    def _simulate(_name: str, *, duration: float, current: float) -> dict[str, object]:
        calls["simulate"] += 1
        return {"spikes": [1, 2], "n_steps": 10, "dt": 0.1, "spike_count": 2}

    monkeypatch.setattr(model_scan, "simulate_model", _simulate)
    monkeypatch.setattr(
        model_scan,
        "classify_firing_pattern",
        lambda spikes, n_steps, dt: {
            "pattern": "tonic",
            "description": "ok",
            "rate_hz": float(len(spikes)),
        },
    )

    first = model_scan.scan_all_models(current=10.0, duration=100.0)
    second = model_scan.scan_all_models(current=10.0, duration=100.0)

    assert first == second
    assert calls["simulate"] == 1
    models = _list_field(first, "models")
    first_model = cast(dict[str, JsonValue], models[0])
    metadata = _object_field(first, "scan_metadata")
    assert first["schema_version"] == model_scan.STUDIO_MODEL_SCAN_SCHEMA_VERSION
    assert first_model["name"] == "ModelA"
    assert first_model["pattern"] == "tonic"
    assert metadata == {
        "current": 10.0,
        "duration": 100.0,
        "evidence_classification": "analysis",
        "input_sha256": metadata["input_sha256"],
        "model_count": 1,
        "pattern_counts": {"tonic": 1},
        "result_sha256": metadata["result_sha256"],
        "schema_version": model_scan.STUDIO_MODEL_SCAN_SCHEMA_VERSION,
        "status": "completed",
    }


def test_scan_all_models_cache_is_keyed_by_scan_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scan cannot reuse cached evidence from another current or duration."""

    model_scan._CACHE.clear()
    calls: list[tuple[float, float]] = []

    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [{"name": "ModelA", "category": "CatA"}],
    )

    def _simulate(_name: str, *, duration: float, current: float) -> dict[str, object]:
        calls.append((current, duration))
        return {"spikes": [1, 2], "n_steps": 10, "dt": 0.1, "spike_count": 2}

    monkeypatch.setattr(model_scan, "simulate_model", _simulate)
    monkeypatch.setattr(
        model_scan,
        "classify_firing_pattern",
        lambda spikes, n_steps, dt: {
            "pattern": "tonic",
            "description": "ok",
            "rate_hz": float(len(spikes)),
        },
    )

    first = model_scan.scan_all_models(current=10.0, duration=100.0)
    second = model_scan.scan_all_models(current=20.0, duration=100.0)
    first_metadata = _object_field(first, "scan_metadata")
    second_metadata = _object_field(second, "scan_metadata")

    assert calls == [(10.0, 100.0), (20.0, 100.0)]
    assert first_metadata["current"] == 10.0
    assert second_metadata["current"] == 20.0
    assert first_metadata["input_sha256"] != second_metadata["input_sha256"]


def test_scan_all_models_fails_closed_with_structured_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_scan._CACHE.clear()

    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [
            {"name": "GoodModel", "category": "Cortex"},
            {"name": "BadModel", "category": "Cortex"},
        ],
    )

    def _simulate(name: str, *, duration: float, current: float) -> dict[str, object]:
        if name == "BadModel":
            raise RuntimeError("backend missing")
        return {"spikes": [1], "n_steps": 10, "dt": 0.1, "spike_count": 1}

    monkeypatch.setattr(model_scan, "simulate_model", _simulate)
    monkeypatch.setattr(
        model_scan,
        "classify_firing_pattern",
        lambda spikes, n_steps, dt: {
            "pattern": "single_spike",
            "description": "ok",
            "rate_hz": 1.0,
        },
    )

    with pytest.raises(ValueError) as exc_info:
        model_scan.scan_all_models(current=10.0, duration=100.0)

    assert "model scan failed for 1/2 models" in str(exc_info.value)
    diagnostics = exc_info.value.args[1]
    assert diagnostics["failed_count"] == 1
    assert diagnostics["total_models"] == 2
    assert diagnostics["failure_rate"] == pytest.approx(0.5)
    assert diagnostics["failed_models"] == [
        {
            "name": "BadModel",
            "category": "Cortex",
            "error_type": "RuntimeError",
            "error_message": "backend missing",
        }
    ]
