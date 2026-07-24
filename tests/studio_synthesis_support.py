# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Synthesis Dashboard (Block 3)

from __future__ import annotations

import json

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform import StudioRuntimeSettings

from sc_neurocore.studio.synthesis import (
    _DEVICE_CAPACITY,
    _TARGETS,
    _parse_yosys_json,
    check_tools,
    estimate_resources,
    multi_target_synthesis,
    run_synthesis,
)


@pytest.fixture(scope="module")
def client():
    return TestClient(create_app(), base_url="http://127.0.0.1")


@pytest.fixture(scope="module")
def large_body_client():
    settings = StudioRuntimeSettings(max_request_body_bytes=4 * 1024 * 1024)
    return TestClient(create_app(runtime_settings=settings), base_url="http://127.0.0.1")


__all__ = [
    "annotations",
    "json",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "StudioRuntimeSettings",
    "_DEVICE_CAPACITY",
    "_TARGETS",
    "_parse_yosys_json",
    "check_tools",
    "estimate_resources",
    "multi_target_synthesis",
    "run_synthesis",
    "client",
    "large_body_client",
]
