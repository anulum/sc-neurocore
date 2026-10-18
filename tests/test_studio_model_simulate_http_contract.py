# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio model-run contract (HTTP surface)

"""HTTP contract of ``POST /api/models/simulate`` and ``POST /api/export/svg``.

A rejected request returns 422 with a structured ``invalid_model_input`` detail,
a numerical failure returns 422 with ``model_simulation_failed``, neither is
cached, and the OpenAPI document declares that contract. Every case fails on the
former behaviour, which returned 200 with substituted inputs or zeroed traces.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

ATIF = "AdaptiveThresholdIFNeuron"
SIMULATE = "/api/models/simulate"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Serve the production HTTP routers throughout the module's request contracts."""
    with TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as test_client:
        yield test_client


def _cache_size(client: TestClient) -> int:
    response = client.get("/api/cache/stats")
    assert response.status_code == 200
    return int(response.json()["size"])


def _post(client: TestClient, route: str, body: dict[str, Any]) -> Any:
    return client.post(route, json=body)


def _raw_post(client: TestClient, route: str, raw_body: bytes) -> Any:
    return client.post(route, content=raw_body, headers={"Content-Type": "application/json"})


class TestRejectedRequests:
    """Expose schema and model-domain refusals through the actual simulation HTTP routes."""

    def test_unknown_model_name(self, client: TestClient) -> None:
        """Return the catalogue name refusal as a structured HTTP 422 response."""
        response = _post(client, SIMULATE, {"model_name": "NoSuchNeuron"})
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "error": "invalid_model_input",
            "model": "NoSuchNeuron",
            "field": "name",
            "reason": "unknown model",
        }

    def test_unknown_parameter(self, client: TestClient) -> None:
        """Keep the rejected constructor keyword in the public HTTP refusal."""
        response = _post(
            client, SIMULATE, {"model_name": ATIF, "params": {"no_such_parameter": 1.0}}
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["error"] == "invalid_model_input"
        assert (detail["model"], detail["field"]) == (ATIF, "params.no_such_parameter")

    def test_invalid_constructor(self, client: TestClient) -> None:
        """Keep the constructor refusal structured without copying its ValueError."""
        response = _post(client, SIMULATE, {"model_name": ATIF, "params": {"theta_rest": -70.0}})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["field"] == "constructor"
        assert detail["reason"] == "model constructor rejected the supplied parameters"

    def test_fractional_integer_parameter(self, client: TestClient) -> None:
        """Reject a fractional threshold with its model parameter field named."""
        response = _post(
            client,
            SIMULATE,
            {"model_name": "IntegerQIFNeuron", "params": {"v_threshold": 30.5}},
        )
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["field"] == "params.v_threshold"
        assert "fractional" in detail["reason"]

    @pytest.mark.parametrize("value", [10**400, -(10**400)])
    def test_integer_outside_float64_has_a_structured_field_refusal(
        self, client: TestClient, value: int
    ) -> None:
        """Reject a finite integer outside Float64 without an unhandled conversion fault."""
        response = _post(client, SIMULATE, {"model_name": ATIF, "params": {"tau_m": value}})
        assert response.status_code == 422
        assert response.json()["detail"] == {
            "error": "invalid_model_input",
            "model": ATIF,
            "field": "params.tau_m",
            "reason": "must be a finite number",
        }

    def test_step_requiring_unsupplied_input(self, client: TestClient) -> None:
        """Return the missing glutamate argument as a step-contract refusal."""
        response = _post(client, SIMULATE, {"model_name": "DendriticNMDANeuron"})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["field"] == "step"
        assert "glutamate" in detail["reason"]

    @pytest.mark.parametrize(
        ("body", "loc_tail", "error_type"),
        [
            (
                {"model_name": ATIF, "params": {"tau_m": True}},
                ["params", "tau_m", "float"],
                "float_type",
            ),
            (
                {"model_name": ATIF, "params": {"tau_m": "10"}},
                ["params", "tau_m", "float"],
                "float_type",
            ),
            ({"model_name": ATIF, "dt": 0.0}, ["dt"], "greater_than"),
            ({"model_name": ATIF, "duration": -1.0}, ["duration"], "greater_than"),
            ({"model_name": ATIF, "protocol": "sawtooth"}, ["protocol"], "literal_error"),
            ({"model_name": ATIF, "use_fast_path": False}, ["use_fast_path"], "extra_forbidden"),
        ],
    )
    def test_body_schema_rejects_non_numeric_and_unknown_fields(
        self,
        client: TestClient,
        body: dict[str, Any],
        loc_tail: list[str],
        error_type: str,
    ) -> None:
        """Keep the typed numeric-branch error attached to the original parameter field."""
        response = _post(client, SIMULATE, body)
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert isinstance(errors, list)
        assert any(
            error["type"] == error_type and list(error["loc"])[-len(loc_tail) :] == loc_tail
            for error in errors
        ), errors

    @pytest.mark.parametrize(
        ("raw_body", "loc_tail"),
        [
            (b'{"model_name": "%s", "dt": NaN}', ["dt"]),
            (b'{"model_name": "%s", "params": {"tau_m": Infinity}}', ["params", "tau_m", "float"]),
            (b'{"model_name": "%s", "current": -Infinity}', ["current"]),
        ],
    )
    def test_non_finite_json_number_is_422_not_500(
        self, client: TestClient, raw_body: bytes, loc_tail: list[str]
    ) -> None:
        """Reject non-finite JSON tokens with a numeric field and validation type."""
        response = _raw_post(client, SIMULATE, raw_body % ATIF.encode())
        assert response.status_code == 422
        errors = response.json()["detail"]
        assert isinstance(errors, list)
        assert any(
            error["type"] == "finite_number" and list(error["loc"])[-len(loc_tail) :] == loc_tail
            for error in errors
        ), errors

    def test_export_svg_shares_the_contract(self, client: TestClient) -> None:
        """Reject non-finite and unknown SVG constructor inputs before rendering any trace."""
        response = _raw_post(
            client,
            "/api/export/svg",
            b'{"model_name": "%s", "params": {"tau_m": Infinity}}' % ATIF.encode(),
        )
        assert response.status_code == 422
        assert isinstance(response.json()["detail"], list)
        response = _post(client, "/api/export/svg", {"model_name": ATIF, "params": {"nope": 1.0}})
        assert response.status_code == 422
        assert response.json()["detail"]["field"] == "params.nope"

    def test_fixed_step_model_rejects_foreign_dt(self, client: TestClient) -> None:
        """Refuse a foreign clock and publish the model's actual default step interval."""
        response = _post(client, SIMULATE, {"model_name": "IntegerQIFNeuron", "dt": 0.1})
        assert response.status_code == 422
        detail = response.json()["detail"]
        assert detail["field"] == "dt"
        assert "fixed step of 1.0 ms" in detail["reason"]
        response = _post(client, SIMULATE, {"model_name": "IntegerQIFNeuron", "duration": 20.0})
        assert response.status_code == 200
        assert response.json()["effective_inputs"]["dt_source"] == "model_attribute"
        assert response.json()["dt"] == 1.0


class TestNumericalFailure:
    """Keep numerical and admission failures out of the response cache."""

    OVERFLOW = {
        "model_name": "HodgkinHuxleyNeuron",
        "protocol": "ramp",
        "current": 1e300,
        "duration": 5.0,
        "dt": 0.01,
    }

    def test_intermediate_overflow_is_422_with_step_and_never_cached(
        self, client: TestClient
    ) -> None:
        """A numerical fault keeps its step metadata and never populates the cache."""
        size_before = _cache_size(client)
        first = _post(client, SIMULATE, self.OVERFLOW)
        assert first.status_code == 422
        detail = first.json()["detail"]
        assert detail["error"] == "model_simulation_failed"
        assert (detail["model"], detail["backend"]) == ("HodgkinHuxleyNeuron", "python")
        assert detail["step"] > 0
        # A Hodgkin-Huxley step is a 1 ms macro step of 0.01 ms sub-steps.
        assert detail["time_ms"] == pytest.approx(detail["step"] * 1.0)
        assert detail["diagnostic"] == "model step could not produce a finite result"
        assert "time" not in first.json()
        assert _cache_size(client) == size_before
        second = _post(client, SIMULATE, self.OVERFLOW)
        assert second.status_code == 422
        assert second.json()["detail"] == detail

    def test_rejected_request_is_never_cached(self, client: TestClient) -> None:
        """Leave cache size unchanged after a rejected unknown constructor parameter."""
        size_before = _cache_size(client)
        body = {"model_name": ATIF, "params": {"unknown_parameter": 2.0}}
        assert _post(client, SIMULATE, body).status_code == 422
        assert _cache_size(client) == size_before


class TestSuccessfulRun:
    """Publish effective constructor inputs and alias-equivalent model runs."""

    def test_valid_nondefault_adaptive_threshold_run(self, client: TestClient) -> None:
        """Report nondefault constructor inputs beside the actual complete voltage trace."""
        body = {
            "model_name": ATIF,
            "params": {"delta_theta": 8.0, "tau_theta": 30.0, "theta_rest": -48.0},
            "current": 20.0,
            "duration": 50.0,
        }
        response = _post(client, SIMULATE, body)
        assert response.status_code == 200
        data = response.json()
        receipt = data["effective_inputs"]
        assert receipt["backend"] == "python"
        assert receipt["overrides_applied"] == ["delta_theta", "tau_theta", "theta_rest"]
        assert receipt["parameters"]["theta_rest"] == -48.0
        assert receipt["parameters"]["tau_m"] == 10.0
        assert receipt["state_recording"] == {"recorded": ["v", "theta"], "excluded": []}
        assert data["spike_count"] > 0
        assert len(data["states"]["v"]) == data["n_steps"] == 500
        assert "pattern" in data
        assert data["run_metadata"]["source"] == "model"

    def test_model_name_and_name_aliases_are_equivalent(self, client: TestClient) -> None:
        """Resolve both public model name spellings to the same spike trace."""
        by_alias = _post(client, SIMULATE, {"model_name": ATIF, "duration": 5.0})
        by_name = _post(client, SIMULATE, {"name": ATIF, "duration": 5.0})
        assert by_alias.status_code == by_name.status_code == 200
        assert by_alias.json()["spikes"] == by_name.json()["spikes"]


class TestOpenApiContract:
    """Declare simulation refusals and request restrictions in the generated OpenAPI document."""

    def test_simulate_route_declares_the_error_response(self, client: TestClient) -> None:
        """Expose structured 422 schemas, forbidden unknown fields and step metadata."""
        document = client.get("/openapi.json").json()
        for route in (SIMULATE, "/api/export/svg"):
            operation = document["paths"][route]["post"]
            schema = operation["responses"]["422"]["content"]["application/json"]["schema"]
            assert schema["$ref"].endswith("/ModelRunErrorResponse")
        components = document["components"]["schemas"]
        assert components["ModelSimulateRequest"]["additionalProperties"] is False
        assert components["ModelInputErrorDetail"]["properties"]["error"]["const"] == (
            "invalid_model_input"
        )
        assert components["ModelSimulationFailureDetail"]["properties"]["error"]["const"] == (
            "model_simulation_failed"
        )
        assert set(components["ModelSimulationFailureDetail"]["required"]) == {
            "error",
            "model",
            "backend",
            "step",
            "time_ms",
            "diagnostic",
        }
