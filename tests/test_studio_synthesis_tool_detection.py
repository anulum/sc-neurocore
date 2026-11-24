# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis tool detection

"""Focused suite: TestToolDetection from former test_studio_synthesis.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

from tests.studio_synthesis_support import (
    check_tools,
)
from tests.studio_synthesis_support import client as client


class TestToolDetection:
    """Detection of the EDA tools."""

    def test_check_tools_returns_dict(self) -> None:
        """Tool detection reports availability and version for Yosys and nextpnr."""
        result = check_tools()
        assert "yosys" in result
        assert "nextpnr_ice40" in result
        for tool_info in result.values():
            assert "available" in tool_info
            assert "version" in tool_info

    def test_check_tools_has_all_expected(self) -> None:
        """Tool detection reports exactly the expected tools."""
        result = check_tools()
        expected = {"yosys", "nextpnr_ice40", "nextpnr_ecp5", "firtool"}
        assert set(result.keys()) == expected

    def test_tools_status_endpoint(self, client: TestClient) -> None:
        """The tools-status endpoint reports Yosys availability as a boolean."""
        r = client.get("/api/synth/tools-status")
        assert r.status_code == 200
        data = r.json()
        assert "yosys" in data
        assert isinstance(data["yosys"]["available"], bool)
