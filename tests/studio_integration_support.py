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

from collections.abc import Iterator

from pathlib import Path

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.network_graph import create_population, create_projection

from sc_neurocore.studio.project import (
    PIPELINE_ROUTE,
    delete_project,
    list_projects,
    load_project,
    run_pipeline,
    save_project,
)


@pytest.fixture(scope="module")
def client(tmp_path_factory: pytest.TempPathFactory) -> Iterator[TestClient]:
    """A Studio client whose saved workspaces live under a temporary root.

    The default project root is the user's home directory. A case that saved
    into it left a workspace behind, so the next run saved over a workspace it
    had never loaded — which the revision store correctly refuses. Tests do not
    write to the person's real Studio.
    """

    root = tmp_path_factory.mktemp("studio-projects")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("sc_neurocore.studio.project._PROJECTS_DIR", str(root))
        yield TestClient(create_app(), base_url="http://127.0.0.1")


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
    "PIPELINE_ROUTE",
    "run_pipeline",
    "save_project",
    "client",
]
