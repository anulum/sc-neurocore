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
        "error_count": 0,
        "evidence_classification": "analysis",
        "failed_models": [],
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


def test_scan_survives_real_undriveable_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real simulation path stays resilient to a genuinely undriveable model.

    ``DendriticNMDANeuron`` needs a synaptic glutamate input its ``step`` cannot
    get from a constant current. The run contract rejects it before any step as
    a ``ModelInputError`` naming the missing input. Here the genuine
    ``simulate_model`` runs (only the model list is narrowed) and the scan must
    still classify the drivable model and report the structured failure.
    """

    model_scan._CACHE.clear()
    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [
            {"name": "ThetaNeuron", "category": "Integrate-and-Fire"},
            {"name": "DendriticNMDANeuron", "category": "Synaptic"},
        ],
    )

    scan = model_scan.scan_all_models(current=10.0, duration=100.0)

    models = _list_field(scan, "models")
    by_name = {cast(dict[str, JsonValue], m)["name"]: cast(dict[str, JsonValue], m) for m in models}
    assert by_name["ThetaNeuron"]["pattern"] != "error"
    assert by_name["DendriticNMDANeuron"]["pattern"] == "error"
    assert by_name["DendriticNMDANeuron"]["error_type"] == "ModelInputError"
    assert "glutamate" in str(by_name["DendriticNMDANeuron"]["description"])

    metadata = _object_field(scan, "scan_metadata")
    assert metadata["error_count"] == 1
    failed = {cast(dict[str, JsonValue], f)["name"] for f in _list_field(metadata, "failed_models")}
    assert failed == {"DendriticNMDANeuron"}


def test_scan_all_models_reports_per_model_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A model that cannot be driven is reported, not allowed to abort the scan.

    The scan must complete for the rest of the catalogue and surface the failure
    explicitly (an ``error`` entry plus ``error_count`` / ``failed_models``), so a
    single undriveable model never breaks the whole endpoint.
    """

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

    scan = model_scan.scan_all_models(current=10.0, duration=100.0)

    models = _list_field(scan, "models")
    by_name = {cast(dict[str, JsonValue], m)["name"]: cast(dict[str, JsonValue], m) for m in models}
    assert by_name["GoodModel"]["pattern"] == "single_spike"
    assert by_name["BadModel"]["pattern"] == "error"
    assert by_name["BadModel"]["error_type"] == "RuntimeError"
    assert "backend missing" in str(by_name["BadModel"]["description"])

    metadata = _object_field(scan, "scan_metadata")
    assert metadata["error_count"] == 1
    assert metadata["model_count"] == 2
    assert metadata["status"] == "completed"
    assert metadata["failed_models"] == [
        {
            "category": "Cortex",
            "error_message": "RuntimeError: backend missing",
            "error_type": "RuntimeError",
            "name": "BadModel",
        }
    ]
