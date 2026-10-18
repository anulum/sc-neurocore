# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio complete-state custody (HTTP surface)

"""HTTP custody contract of the Studio simulation routes.

``POST /api/models/simulate``, ``POST /api/multi-simulate`` and
``POST /api/simulate`` return the declared state layout with its custody
verdict, exact snapshots, the full-resolution raw block and a bounded display
projection under ``studio.simulation-run.v2`` metadata; the model routes run
on the Python custody backend even when the Rust batch backend is loaded, the
result cache never retains oversized raw blocks, and evidence bundles accept
both manifest versions.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio import model_simulate
from sc_neurocore.studio.api import simulation as simulation_api
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform.evidence_bundle_payloads import _simulation_result_payload
from sc_neurocore.studio.simulation_manifest import (
    STUDIO_SIMULATION_RUN_SCHEMA_V1,
    STUDIO_SIMULATION_RUN_SCHEMA_VERSION,
)

ATIF = "AdaptiveThresholdIFNeuron"


@pytest.fixture(scope="module")
def client() -> Iterator[TestClient]:
    """Provide the actual Studio API for simulation state and result custody."""
    with TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    ) as test_client:
        yield test_client


class TestModelSimulateCustody:
    """Raw state, snapshots and bounded display traces across simulation routes."""

    def test_response_carries_layout_snapshots_raw_and_display(self, client: TestClient) -> None:
        """Retain state layout and snapshots beside raw and display recordings."""
        response = client.post(
            "/api/models/simulate",
            json={"model_name": ATIF, "current": 20.0, "duration": 50.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state_layout"]["recorded"] == ["v", "theta"]
        assert data["state_layout"]["complete"] is True
        assert data["observation"]["clock"] == "post-step"
        assert set(data["initial_state"]) == set(data["final_state"]) == {"v", "theta"}
        assert data["raw"]["included"] is True
        assert len(data["raw"]["states"]["theta"]) == data["n_steps"] == 500
        assert data["display"]["method"] == "identity"
        assert data["time"][0] == pytest.approx(0.1)
        metadata = data["run_metadata"]
        assert metadata["schema_version"] == STUDIO_SIMULATION_RUN_SCHEMA_VERSION
        assert metadata["state_variables"] == ["theta", "v"]
        assert metadata["layout_source"] == "descriptor"
        assert metadata["state_custody_complete"] is True
        assert metadata["raw_included"] is True
        assert metadata["observation_clock"] == "post-step"

    def test_model_route_uses_the_python_custody_backend(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Declare the Python backend actually used for model-state custody."""

        def _never() -> None:
            raise AssertionError("the custody route must not use the Rust batch backend")

        monkeypatch.setattr(model_simulate, "_load_rust_batch_simulate", _never)
        response = client.post(
            "/api/models/simulate", json={"model_name": "AdExNeuron", "duration": 3.0}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["effective_inputs"]["backend"] == "python"
        assert data["state_layout"]["recorded"] == ["v", "w"]
        assert data["initial_state"] is not None

    def test_long_run_is_bounded_on_display_and_complete_in_raw(self, client: TestClient) -> None:
        """Retain complete raw recordings while bounding the displayed samples."""
        response = client.post(
            "/api/models/simulate",
            json={"model_name": ATIF, "current": 20.0, "duration": 999.9},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["n_steps"] == 9_999
        assert data["display"]["point_count"] <= 5_000
        assert len(data["time"]) == data["display"]["point_count"]
        assert len(data["raw"]["states"]["v"]) == 9_999
        assert data["display"]["sample_index"][-1] == 9_998
        assert data["run_metadata"]["sample_count"] == data["display"]["point_count"]

    def test_multi_simulate_shares_the_custody_contract(self, client: TestClient) -> None:
        """Preserve the same state-custody contract in multi-model simulation."""
        response = client.post(
            "/api/multi-simulate",
            json=[{"name": ATIF, "duration": 5.0}, {"name": "AdExNeuron", "duration": 5.0}],
        )
        assert response.status_code == 200
        for result in response.json():
            assert result["effective_inputs"]["backend"] == "python"
            assert result["raw"]["included"] is True
            assert result["run_metadata"]["schema_version"] == STUDIO_SIMULATION_RUN_SCHEMA_VERSION

    def test_ode_route_shares_the_custody_contract(self, client: TestClient) -> None:
        """Declare equation-state layout through the shared simulation response."""
        response = client.post(
            "/api/simulate",
            json={
                "equations": ["dv/dt = I"],
                "init": {"v": 0.0},
                "dt": 0.1,
                "duration": 1.0,
                "current": 1.0,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["state_layout"]["source"] == "equations"
        assert data["initial_state"] == {"v": 0.0}
        assert data["raw"]["states"]["v"][-1] == pytest.approx(1.0)

    def test_svg_export_uses_the_effective_step(self, client: TestClient) -> None:
        """Render SVG output using the simulation's effective time step."""
        response = client.post(
            "/api/export/svg", json={"model_name": "IntegerQIFNeuron", "duration": 20.0}
        )
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("image/svg+xml")


class TestCacheGuard:
    """Cache admission for complete raw recordings and legacy result payloads."""

    def test_oversized_raw_results_are_not_retained(self) -> None:
        """Refuse cache retention when the raw recording exceeds its byte budget."""
        cache = simulation_api._SimCache(maxsize=4)
        small = {"raw": {"element_count": 10}, "time": [0.1]}
        big = {"raw": {"element_count": simulation_api.CACHE_RAW_ELEMENT_LIMIT + 1}, "time": [0.1]}
        cache.put({"k": "small"}, small)
        cache.put({"k": "big"}, big)
        assert cache.get({"k": "small"}) is small
        assert cache.get({"k": "big"}) is None
        assert len(cache._cache) == 1

    def test_results_without_a_raw_block_are_retained(self) -> None:
        """Retain supported legacy results which have no raw-recording block."""
        cache = simulation_api._SimCache(maxsize=4)
        legacy: dict[str, Any] = {"time": [0.1]}
        cache.put({"k": "legacy"}, legacy)
        assert cache.get({"k": "legacy"}) is legacy


class TestEvidenceBundleVersions:
    """Supported simulation manifests at the evidence-bundle input boundary."""

    @pytest.mark.parametrize(
        "version", [STUDIO_SIMULATION_RUN_SCHEMA_V1, STUDIO_SIMULATION_RUN_SCHEMA_VERSION]
    )
    def test_both_manifest_versions_are_accepted(self, version: str) -> None:
        """Accept both explicitly supported simulation-result manifest versions."""
        payload = {
            "time": [0.1],
            "states": {"v": [0.0]},
            "run_metadata": {
                "schema_version": version,
                "evidence_classification": "simulation",
                "status": "completed",
            },
        }
        assert _simulation_result_payload(payload)["time"] == [0.1]

    def test_unknown_manifest_version_is_rejected(self) -> None:
        """Refuse a simulation-result manifest with an undeclared version."""
        payload = {
            "run_metadata": {
                "schema_version": "studio.simulation-run.v3",
                "evidence_classification": "simulation",
                "status": "completed",
            }
        }
        with pytest.raises(ValueError, match="unsupported run metadata"):
            _simulation_result_payload(payload)
