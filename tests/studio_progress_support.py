# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio WebSocket progress streaming

from __future__ import annotations

import hashlib

import json

import queue

from pathlib import Path

from typing import Any

import pytest

fastapi = pytest.importorskip("fastapi")

from starlette.testclient import TestClient

from starlette.websockets import WebSocketDisconnect

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.platform import Principal, StudioRuntimeSettings

from sc_neurocore.studio.progress import (
    _characterize_with_progress,
    _heatmap_with_progress,
    _scan_with_progress,
)

@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(
        create_app(runtime_settings=StudioRuntimeSettings(allowed_hosts=("testserver",))),
        headers={"origin": "http://127.0.0.1:8001"},
    )

__all__ = [
    "annotations",
    "hashlib",
    "json",
    "queue",
    "Path",
    "Any",
    "pytest",
    "fastapi",
    "TestClient",
    "WebSocketDisconnect",
    "create_app",
    "Principal",
    "StudioRuntimeSettings",
    "_characterize_with_progress",
    "_heatmap_with_progress",
    "_scan_with_progress",
    "client",
]
