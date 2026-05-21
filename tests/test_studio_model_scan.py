# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio model scan hardening and cache behaviour

from __future__ import annotations

import pytest

from sc_neurocore.studio import model_scan


def test_scan_all_models_caches_successful_results(monkeypatch: pytest.MonkeyPatch) -> None:
    model_scan._CACHE = None
    calls: dict[str, int] = {"simulate": 0}

    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [{"name": "ModelA", "category": "CatA"}],
    )

    def _simulate(_name: str, *, duration: float, current: float):
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
    assert first[0]["name"] == "ModelA"
    assert first[0]["pattern"] == "tonic"


def test_scan_all_models_fails_closed_with_structured_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_scan._CACHE = None

    monkeypatch.setattr(
        model_scan,
        "list_models",
        lambda: [
            {"name": "GoodModel", "category": "Cortex"},
            {"name": "BadModel", "category": "Cortex"},
        ],
    )

    def _simulate(name: str, *, duration: float, current: float):
        if name == "BadModel":
            raise RuntimeError("backend missing")
        return {"spikes": [1], "n_steps": 10, "dt": 0.1, "spike_count": 1}

    monkeypatch.setattr(model_scan, "simulate_model", _simulate)
    monkeypatch.setattr(
        model_scan,
        "classify_firing_pattern",
        lambda spikes, n_steps, dt: {"pattern": "single_spike", "description": "ok", "rate_hz": 1.0},
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
