# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio model catalogue test support

"""Shared imports and fixtures for model catalogue tests."""

from __future__ import annotations

from typing import cast

import pytest
from starlette.testclient import TestClient

from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.models import (
    _introspected_summary,
    get_model_detail,
    list_models,
    model_documentation,
    model_facets,
)

__all__ = [
    "TestClient",
    "_adex_summary",
    "_introspected_summary",
    "cast",
    "client",
    "get_model_detail",
    "list_models",
    "model_documentation",
    "model_facets",
]


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(), base_url="http://127.0.0.1")


def _adex_summary() -> dict[str, object]:
    return next(m for m in list_models() if m["name"] == "AdExNeuron")
