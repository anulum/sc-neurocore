# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio endpoint coverage (network, analysis, codegen)

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform import (
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS,
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION,
    DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS,
    StudioRuntimeSettings,
)

MODEL = "AdExNeuron"


@pytest.fixture(scope="module")
def client():
    app = create_app()
    return TestClient(app, base_url="http://127.0.0.1")


def _budget_client(
    *,
    max_sync_analysis_steps_per_simulation: int = (
        DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION
    ),
    max_sync_analysis_total_steps: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS,
    max_sync_analysis_simulations: int = DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS,
) -> TestClient:
    """Shared fixtures for Studio HTTP endpoint contract tests."""
    settings = StudioRuntimeSettings(
        max_sync_analysis_steps_per_simulation=max_sync_analysis_steps_per_simulation,
        max_sync_analysis_total_steps=max_sync_analysis_total_steps,
        max_sync_analysis_simulations=max_sync_analysis_simulations,
    )
    return TestClient(create_app(settings), base_url="http://127.0.0.1")


__all__ = [
    "annotations",
    "pytest",
    "fastapi",
    "httpx",
    "TestClient",
    "create_app",
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_SIMULATIONS",
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_STEPS_PER_SIMULATION",
    "DEFAULT_STUDIO_MAX_SYNC_ANALYSIS_TOTAL_STEPS",
    "StudioRuntimeSettings",
    "MODEL",
    "client",
    "_budget_client",
]
