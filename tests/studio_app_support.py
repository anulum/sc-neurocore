# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio FastAPI backend

"""Provide real Studio HTTP fixtures for application and template contracts."""

from __future__ import annotations

import logging

import pytest

fastapi = pytest.importorskip("fastapi")

httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.templates import TEMPLATES


@pytest.fixture
def client() -> TestClient:
    """Provide the real Studio API for application and template contracts."""
    app = create_app()
    return TestClient(app, base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"})


__all__ = [
    "annotations",
    "logging",
    "pytest",
    "fastapi",
    "httpx",
    "TestClient",
    "create_app",
    "TEMPLATES",
    "client",
]
