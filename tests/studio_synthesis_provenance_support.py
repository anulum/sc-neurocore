# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_studio_synthesis_provenance.py

from __future__ import annotations

"""Tests for Studio synthesis target provenance contracts."""

import hashlib

import json

import re

from typing import cast

import pytest

from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app

from sc_neurocore.studio.synthesis import run_synthesis

from sc_neurocore.studio.synthesis_provenance import (
    STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION,
    STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION,
    JsonValue,
    StudioSynthesisTargetProvenance,
    StudioSynthesisToolProvenance,
    ToolStatusMap,
    build_synthesis_target_provenance,
    build_synthesis_target_provenance_matrix,
)

@pytest.fixture
def client() -> TestClient:
    """Return a Studio test client."""

    return TestClient(create_app(), base_url="http://127.0.0.1")

def _tool_status() -> ToolStatusMap:
    """Return deterministic path-free EDA tool status."""

    return cast(
        ToolStatusMap,
        {
            "yosys": {"available": True, "version": "Yosys 0.test"},
            "nextpnr_ice40": {"available": False, "version": None},
            "nextpnr_ecp5": {"available": True, "version": "nextpnr-ecp5 test"},
            "firtool": {"available": False, "version": 123},
        },
    )


__all__ = ['hashlib', 'json', 're', 'cast', 'pytest', 'TestClient', 'create_app', 'run_synthesis', 'STUDIO_SYNTHESIS_TARGET_PROVENANCE_MATRIX_SCHEMA_VERSION', 'STUDIO_SYNTHESIS_TARGET_PROVENANCE_SCHEMA_VERSION', 'JsonValue', 'StudioSynthesisTargetProvenance', 'StudioSynthesisToolProvenance', 'ToolStatusMap', 'build_synthesis_target_provenance', 'build_synthesis_target_provenance_matrix', 'client', '_tool_status']
