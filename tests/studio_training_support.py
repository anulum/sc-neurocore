# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio Training Monitor (Block 4)

from __future__ import annotations

import json

import threading

import time

from pathlib import Path

from typing import cast

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform import (
    STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION,
    StudioRuntimeSettings,
)

from sc_neurocore.studio.platform.jobs import (
    StudioJobCancelled,
    StudioJobContext,
    StudioJobManager,
)

from sc_neurocore.studio.training import (
    TrainingJob,
    _CELL_TYPES,
    _SURROGATES,
    _register_job,
    get_training_status,
    list_cell_types,
    list_jobs,
    list_surrogates,
    start_training,
    stop_training,
    stream_metrics,
)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


__all__ = [
    "annotations",
    "json",
    "threading",
    "time",
    "Path",
    "cast",
    "pytest",
    "fastapi",
    "TestClient",
    "create_app",
    "STUDIO_ACTION_EVIDENCE_SCHEMA_VERSION",
    "StudioRuntimeSettings",
    "StudioJobCancelled",
    "StudioJobContext",
    "StudioJobManager",
    "TrainingJob",
    "_CELL_TYPES",
    "_SURROGATES",
    "_register_job",
    "get_training_status",
    "list_cell_types",
    "list_jobs",
    "list_surrogates",
    "start_training",
    "stop_training",
    "stream_metrics",
    "client",
]
