# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio training listing

"""Focused suite: TestListing from former test_studio_training.py."""

from __future__ import annotations

from tests.studio_training_support import (
    TestClient,
    _CELL_TYPES,
    _SURROGATES,
    list_cell_types,
    list_surrogates,
)
from tests.studio_training_support import client as client


class TestListing:
    """Listing the training surrogates and cell types."""

    def test_list_surrogates(self) -> None:
        """Every surrogate is listed, atan and fast sigmoid among them."""
        result = list_surrogates()
        assert len(result) == len(_SURROGATES)
        names = {s["name"] for s in result}
        assert "atan_surrogate" in names
        assert "fast_sigmoid" in names

    def test_list_cell_types(self) -> None:
        """Every cell type is listed, LIF and AdEx among them."""
        result = list_cell_types()
        assert len(result) == len(_CELL_TYPES)
        names = {c["name"] for c in result}
        assert "LIFCell" in names
        assert "AdExCell" in names

    def test_surrogates_endpoint(self, client: TestClient) -> None:
        """The surrogates endpoint lists every surrogate with its availability."""
        r = client.get("/api/training/surrogates")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_SURROGATES)
        assert all("name" in s for s in data)
        assert all("available" in s for s in data)

    def test_cell_types_endpoint(self, client: TestClient) -> None:
        """The cell-types endpoint lists every cell type."""
        r = client.get("/api/training/cell-types")
        assert r.status_code == 200
        data = r.json()
        assert len(data) == len(_CELL_TYPES)
