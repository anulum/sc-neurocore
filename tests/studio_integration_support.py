# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Integration (Block 6)

from __future__ import annotations

import json

import re

from pathlib import Path

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.network_graph import create_population, create_projection

from sc_neurocore.studio.project import (
    delete_project,
    list_projects,
    load_project,
    run_pipeline,
    save_project,
)

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")

__all__ = [
    "annotations",
    "json",
    "re",
    "Path",
    "Any",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "create_population",
    "create_projection",
    "delete_project",
    "list_projects",
    "load_project",
    "run_pipeline",
    "save_project",
    "client",
]
