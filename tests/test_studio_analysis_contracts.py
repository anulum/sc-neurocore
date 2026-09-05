# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio analysis metric-contract tests

"""Every Studio analysis states its metric contract; undefined points are not zeros."""

from __future__ import annotations

from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.analysis import (
    bifurcation_sweep,
    fi_curve_sweep,
    frequency_response,
    heatmap_2d,
    sensitivity_analysis,
    spike_triggered_average,
)
from sc_neurocore.studio.analysis_contract import (
    METRIC_CONTRACT_SCHEMA_VERSION,
    MetricContract,
    attach_contract,
    contract_summary,
)
from sc_neurocore.studio.analysis_manifest import build_analysis_result_manifest
from sc_neurocore.studio.app import create_app

LIF: dict[str, Any] = {
    "equations": ["dv/dt = -(v - E_L) / tau_m + I / C"],
    "threshold": "v > -50",
    "reset": "v = -65",
    "params": {"E_L": -65.0, "tau_m": 10.0, "C": 1.0},
    "init": {"v": -65.0},
    "dt": 0.1,
    "duration": 50.0,
    "current": 30.0,
}


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _rate_fn(rate_of: Any) -> Any:
    def fake(**cfg: Any) -> dict[str, Any]:
        return {"stats": {"rate_hz": float(rate_of(cfg))}}

    return fake


def test_metric_contract_validates_and_serialises() -> None:
    contract = MetricContract(
        kind="fi-curve",
        definition="rate",
        units={"rates": "Hz"},
        applicability=("constant current",),
        limitations=("one spike per duration",),
        domain="partial",
        domain_detail={"undefined": 1},
    )
    public = contract.to_public_dict()
    assert public["schema_version"] == METRIC_CONTRACT_SCHEMA_VERSION
    assert public["units"] == {"rates": "Hz"} and public["domain"] == "partial"
    payload = attach_contract({"rates": [1.0]}, contract)
    assert contract_summary(payload) == ("fi-curve", "partial")
    assert contract_summary({"rates": [1.0]}) == (None, None)
    with pytest.raises(ValueError, match="kind"):
        MetricContract(kind="", definition="x", units={}, applicability=())
    with pytest.raises(ValueError, match="domain"):
        MetricContract(kind="k", definition="x", units={}, applicability=(), domain="void")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown domain"):
        contract_summary({"contract": {"kind": "k", "domain": "void"}})


def test_manifest_records_contract_kind_and_domain() -> None:
    payload = attach_contract(
        {"rates": [0.0]},
        MetricContract(
            kind="fi-curve", definition="rate", units={}, applicability=(), domain="empty"
        ),
    )
    manifest = build_analysis_result_manifest(
        analysis_type="fi_curve", source="ode", request_payload={}, result_payload=payload
    ).to_public_dict()
    assert manifest["contract"] == "fi-curve" and manifest["domain"] == "empty"
    assert "contract" in list(manifest["output_keys"])  # type: ignore[arg-type]


def test_sensitivity_reports_undefined_points_with_reasons() -> None:
    """A zero parameter and a zero base rate are undefined elasticities, not 0.0."""
    silent = sensitivity_analysis(_rate_fn(lambda cfg: 0.0), {"params": {"a": 1.0}}, ["a"])
    assert silent["base_rate"] == 0.0
    row = silent["sensitivities"][0]
    assert row["sensitivity"] is None and "no spikes" in row["reason"]
    assert silent["contract"]["domain"] == "empty"
    assert silent["contract"]["units"]["sensitivity"] == "dimensionless"

    def rate(cfg: dict[str, Any]) -> float:
        return 10.0 * float(cfg["params"]["a"]) + 5.0

    mixed = sensitivity_analysis(_rate_fn(rate), {"params": {"a": 2.0, "z": 0.0}}, ["z", "a"])
    rows = {row["param"]: row for row in mixed["sensitivities"]}
    assert rows["z"]["sensitivity"] is None and "zero parameter" in rows["z"]["reason"]
    # elasticity = |d rate / d a| · |a| / rate = 10 · 2 / 25 = 0.8
    assert rows["a"]["sensitivity"] == pytest.approx(0.8)
    assert [row["param"] for row in mixed["sensitivities"]] == ["a", "z"]
    assert mixed["contract"]["domain"] == "partial"
    assert mixed["contract"]["domain_detail"] == {"undefined": 1, "total": 2}


def test_fi_curve_and_frequency_response_state_their_rate_definition() -> None:
    fi = fi_curve_sweep(_rate_fn(lambda cfg: cfg["current"]), 0.0, 10.0, 3)
    assert fi["currents"] == [0.0, 5.0, 10.0] and fi["rates"] == [0.0, 5.0, 10.0]
    assert fi["contract"]["kind"] == "fi-curve"
    assert "transient" in fi["contract"]["definition"]
    assert fi["contract"]["units"]["rates"] == "Hz"

    freq = frequency_response(
        _rate_fn(lambda cfg: cfg["frequency_hz"]), {"dt": 0.1}, 1.0, 100.0, 3, 4.0
    )
    assert freq["contract"]["kind"] == "frequency-response"
    assert freq["contract"]["units"]["frequencies_hz"] == "Hz"
    assert freq["amplitude"] == 4.0


def test_bifurcation_sweep_is_labelled_as_a_numerical_extrema_sweep() -> None:
    def simulate(**cfg: Any) -> dict[str, Any]:
        gain = float(cfg["params"]["g"])
        trace = [gain * (1.0 if index % 2 else -1.0) for index in range(40)]
        return {"raw": {"included": True, "states": {"v": trace, "w": [0.0] * 40}}}

    sweep = bifurcation_sweep(simulate, {"params": {}, "protocol": "sine"}, "g", 0.0, 1.0, 3)
    assert sweep["contract"]["kind"] == "numerical-extrema-sweep"
    assert any(
        "not a bifurcation continuation" in item for item in sweep["contract"]["limitations"]
    )
    assert sweep["variable"] == "v" and sweep["protocol"] == "sine"
    assert sweep["attractor_kinds"] == ["fixed_point", "extrema", "extrema"]
    assert sweep["attractors"][0] == [0.0]
    assert sweep["attractors"][2] == [-1.0, 1.0]

    other = bifurcation_sweep(simulate, {"params": {}}, "g", 0.0, 1.0, 2, variable="w")
    assert other["variable"] == "w" and other["attractor_kinds"] == ["fixed_point", "fixed_point"]

    def short(**cfg: Any) -> dict[str, Any]:
        return {"raw": {"included": True, "states": {"v": [0.0, 1.0, 0.0]}}}

    assert bifurcation_sweep(short, {"params": {}}, "g", 0.0, 1.0, 2)["attractor_kinds"] == [
        "insufficient_samples",
        "insufficient_samples",
    ]
    with pytest.raises(ValueError, match="not a state trace"):
        bifurcation_sweep(simulate, {"params": {}}, "g", 0.0, 1.0, 2, variable="missing")


def test_heatmap_and_spike_triggered_average_carry_contracts() -> None:
    heat = heatmap_2d(
        _rate_fn(lambda cfg: cfg["params"]["x"] + cfg["params"]["y"]),
        {"params": {}},
        "x",
        0.0,
        1.0,
        2,
        "y",
        0.0,
        1.0,
        2,
    )
    assert heat["contract"]["kind"] == "rate-map" and heat["failed_points"] == 0

    sta = spike_triggered_average([], [0.0] * 100, [10, 50, 98], 0.1, window_ms=2.0)
    assert sta["n_spikes"] == 2
    assert sta["contract"]["domain"] == "partial"
    assert sta["contract"]["domain_detail"] == {"spikes": 3, "windows_used": 2}
    empty = spike_triggered_average([], [0.0] * 10, [1], 0.1)
    assert empty["average"] == [] and empty["contract"]["domain"] == "empty"


def test_analysis_routes_attach_contract_and_manifest_summary(client: TestClient) -> None:
    fi = client.post(
        "/api/fi-curve", json={**LIF, "i_min": 0.0, "i_max": 20.0, "i_steps": 3}
    ).json()
    assert fi["contract"]["kind"] == "fi-curve"
    assert fi["analysis_metadata"]["contract"] == "fi-curve"
    assert fi["analysis_metadata"]["domain"] == "complete"

    silent = client.post("/api/sensitivity", json={**LIF, "current": 0.0}).json()
    assert silent["analysis_metadata"]["domain"] == "empty"
    assert all(row["sensitivity"] is None for row in silent["sensitivities"])

    sweep = client.post(
        "/api/bifurcation",
        json={
            **LIF,
            "sweep_param": "C",
            "sweep_min": 0.5,
            "sweep_max": 3.0,
            "sweep_steps": 5,
            "variable": "v",
        },
    ).json()
    assert sweep["contract"]["kind"] == "numerical-extrema-sweep"
    assert sweep["variable"] == "v" and len(sweep["attractor_kinds"]) == 5
    assert sweep["analysis_metadata"]["contract"] == "numerical-extrema-sweep"

    freq = client.post("/api/freq-response", json={**LIF, "amplitude": 20.0, "n_freqs": 3}).json()
    assert freq["analysis_metadata"]["contract"] == "frequency-response"
