# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler cosim

"""Focused suite: TestCosim from former test_studio_compiler.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

from tests.studio_compiler_support import LIF_EQ
from tests.studio_compiler_support import client as client


class TestCosim:
    """Float and fixed-point co-simulation of an equation model."""

    def test_cosim_returns_traces(self, client: TestClient) -> None:
        """Co-simulation returns float and fixed results with a non-negative error."""
        r = client.post("/api/ir/cosim", json=LIF_EQ)
        assert r.status_code == 200
        data = r.json()
        assert "float_result" in data
        assert "fixed_result" in data
        assert "error" in data
        assert data["error"]["max_error"] >= 0
