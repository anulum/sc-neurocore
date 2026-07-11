# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio DCLS learnable-delay panel tests

"""Tests for the Studio surface over the DCLS-max learnable-delay tent kernel."""

from __future__ import annotations

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.dcls import (
    dcls_benchmark,
    dcls_forward_parity,
    dcls_kernel_info,
    dcls_tent_profile,
    probe_backends,
)


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def test_kernel_info_carries_verified_provenance_and_contract() -> None:
    info = dcls_kernel_info()
    assert info["provenance"]["doi"] == "10.48550/arXiv.2112.03740"
    assert info["provenance"]["venue"] == "ICLR"
    assert "Masquelier, T." in info["provenance"]["authors"]
    assert info["fixed_point"]["one"] == 256
    assert info["fixed_point"]["weight_format"] == "Q8.8"
    assert "hdl/sc_dcls_tent_kernel.v" in info["rtl_modules"]


def test_tent_profile_is_a_symmetric_triangle_peaked_at_the_centre() -> None:
    # centre tap 2 (512 Q8.8 = 2.0), sigma 3.0
    profile = dcls_tent_profile(centre_q88=512, sigma_q88=768, n_taps=8)
    gates = profile["gates"]
    assert gates[2] == pytest.approx(1.0)  # peak at the learnable centre
    assert gates[1] == pytest.approx(gates[3])  # symmetric about the centre
    assert gates[5] == 0.0  # outside the tent support
    assert profile["centre"] == pytest.approx(2.0)


def test_tent_profile_validates_bounds() -> None:
    with pytest.raises(ValueError):
        dcls_tent_profile(centre_q88=256, sigma_q88=0, n_taps=8)
    with pytest.raises(ValueError):
        dcls_tent_profile(centre_q88=256, sigma_q88=256, n_taps=0)
    with pytest.raises(ValueError):
        dcls_tent_profile(centre_q88=256, sigma_q88=256, n_taps=10_000)


def test_forward_parity_is_bit_exact_across_live_backends() -> None:
    result = dcls_forward_parity(
        spikes=[1, 0, 1, 1, 0, 1, 0, 1],
        weights_q88=[256] * 8,
        centre_q88=512,
        sigma_q88=768,
    )
    assert result["active_tap_count"] == 5
    assert result["bit_exact"] is True
    live = [b for b in result["backends"] if b.get("live") and b.get("available")]
    assert any(b["backend"] == "python" for b in live)
    assert all(b["bit_exact"] for b in live)


def test_julia_is_not_run_in_process_when_torch_is_loaded() -> None:
    # The test suite imports torch, so the Julia probe must be skipped (no crash).
    import torch  # noqa: F401

    julia = next(b for b in probe_backends() if b["backend"] == "julia")
    assert julia["live"] is False


def test_forward_parity_validates_input() -> None:
    with pytest.raises(ValueError):
        dcls_forward_parity(spikes=[], weights_q88=[], centre_q88=256, sigma_q88=256)
    with pytest.raises(ValueError):
        dcls_forward_parity(spikes=[1, 0], weights_q88=[256], centre_q88=256, sigma_q88=256)


def test_benchmark_reports_fastest_first_speedups_with_context() -> None:
    bench = dcls_benchmark()
    assert bench is not None
    assert "i5-11600K" in bench["cpu"]
    assert bench["hardware_measurement_claimed"] is False  # honest: software, not silicon
    speedups = [b["speedup_over_python"] for b in bench["backends"]]
    assert speedups == sorted(speedups, reverse=True)  # fastest measured first
    python = next(b for b in bench["backends"] if b["backend"] == "python")
    assert python["speedup_over_python"] == pytest.approx(1.0)
    assert max(speedups) > 10  # the accelerated backends are an order faster


def test_api_dcls_benchmark_endpoint(client: TestClient) -> None:
    response = client.get("/api/dcls/benchmark")
    assert response.status_code == 200, response.text
    assert response.json()["backends"][0]["speedup_over_python"] > 1.0


def test_api_dcls_info_endpoint(client: TestClient) -> None:
    response = client.get("/api/dcls/info")
    assert response.status_code == 200, response.text
    assert response.json()["provenance"]["venue"] == "ICLR"


def test_api_dcls_evaluate_endpoint(client: TestClient) -> None:
    response = client.post(
        "/api/dcls/evaluate", json={"centre_q88": 512, "sigma_q88": 768, "n_taps": 8}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["profile"]["gates"][2] == pytest.approx(1.0)
    assert body["forward"]["bit_exact"] is True


def test_api_dcls_evaluate_rejects_mismatched_vectors(client: TestClient) -> None:
    response = client.post(
        "/api/dcls/evaluate",
        json={
            "spikes": [1, 0],
            "weights_q88": [256],
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "spikes and weights_q88 must have equal length"
