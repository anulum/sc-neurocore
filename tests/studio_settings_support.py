# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio runtime settings test support

from __future__ import annotations

"""Shared helpers for Studio runtime settings and app security contract tests."""


import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest

UTC = timezone.utc

fastapi = pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.platform import (
    DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES,
    StudioRuntimeSettings,
    build_default_studio_runtime_settings,
)


def _audit_event_hash(row: dict[str, Any]) -> str:
    unsigned_row = dict(row)
    unsigned_row.pop("event_hash", None)
    canonical_row = json.dumps(
        unsigned_row,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical_row).hexdigest()


__all__ = [
    "UTC",
    "fastapi",
    "httpx",
    "_audit_event_hash",
    "json",
    "hashlib",
    "datetime",
    "timezone",
    "Path",
    "Any",
    "UUID",
    "pytest",
    "TestClient",
    "create_app",
    "DEFAULT_STUDIO_JOB_MAX_ARTIFACT_BYTES",
    "StudioRuntimeSettings",
    "build_default_studio_runtime_settings",
]
