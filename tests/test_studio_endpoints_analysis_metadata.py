# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio endpoints analysis metadata

"""Focused suite: TestAnalysisMetadataConsistency from former test_studio_endpoints.py."""

from __future__ import annotations

from tests.studio_endpoints_support import *  # noqa: F403

class TestAnalysisMetadataConsistency:
    def test_characterize_attaches_analysis_metadata(self, client: TestClient) -> None:
        r = client.post(
            "/api/characterize",
            json={"name": MODEL, "dt": 0.5, "duration": 20.0, "current": 10.0},
        )
        assert r.status_code == 200
        metadata = r.json()["analysis_metadata"]
        assert metadata["schema_version"] == "studio.analysis-result.v1"
        assert metadata["analysis_type"] == "characterize"
        assert metadata["source"] == "model"

    def test_multi_simulate_attaches_run_metadata_per_result(self, client: TestClient) -> None:
        r = client.post(
            "/api/multi-simulate",
            json=[
                {"name": MODEL, "duration": 20, "current": 10},
                {"name": "ChayNeuron", "duration": 20, "current": 10},
            ],
        )
        assert r.status_code == 200
        results = r.json()
        assert len(results) == 2
        for result in results:
            assert result["run_metadata"]["schema_version"] == "studio.simulation-run.v1"
            assert result["run_metadata"]["source"] == "model"

