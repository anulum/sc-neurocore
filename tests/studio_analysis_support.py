# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio analysis and new endpoints

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.codegen import classify_firing_pattern, generate_model_script

from sc_neurocore.studio.analysis import frequency_response, heatmap_2d

from sc_neurocore.studio.simulation import _make_current_trace, simulate

@pytest.fixture
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")

__all__ = [
    "annotations",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "classify_firing_pattern",
    "generate_model_script",
    "frequency_response",
    "heatmap_2d",
    "_make_current_trace",
    "simulate",
    "client",
]
