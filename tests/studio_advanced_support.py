# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio advanced features

from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore_engine.studio import get_ei_network_simulator

from sc_neurocore.studio.characterize import characterize_model

from sc_neurocore.studio.codegen import (
    classify_firing_pattern,
    generate_ode_script,
    generate_oneliner,
)

from sc_neurocore.studio.network import simulate_ei_network


@pytest.fixture
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


__all__ = [
    "annotations",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "get_ei_network_simulator",
    "characterize_model",
    "classify_firing_pattern",
    "generate_ode_script",
    "generate_oneliner",
    "simulate_ei_network",
    "client",
]
