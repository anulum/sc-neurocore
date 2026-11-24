# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis estimation

"""Focused suite: TestEstimation from former test_studio_synthesis.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

import pytest
from tests.studio_synthesis_support import (
    _TARGETS,
    estimate_resources,
)
from tests.studio_synthesis_support import client as client


class TestEstimation:
    """Resource estimation for the synthesis targets."""

    def test_estimate_returns_structure(self) -> None:
        """An estimate reports its resources, capacity and utilisation."""
        result = estimate_resources(10, "ice40")
        assert result["target"] == "ice40"
        assert result["estimated"] is True
        assert "resources" in result
        assert "capacity" in result
        assert "utilisation" in result

    def test_estimate_all_targets(self) -> None:
        """Every target gives an estimate with its utilisation keys."""
        for target in _TARGETS:
            result = estimate_resources(5, target)
            assert result["target"] == target
            for k in ["luts", "ffs", "brams", "dsps"]:
                assert k in result["utilisation"]

    def test_estimate_unknown_target_raises(self) -> None:
        """An unknown target raises ValueError."""
        with pytest.raises(ValueError, match="Unknown target"):
            estimate_resources(5, "unknown")

    def test_estimate_scales_with_ops(self) -> None:
        """More operations give more LUTs and flip-flops."""
        small = estimate_resources(5, "ice40")
        large = estimate_resources(50, "ice40")
        assert large["resources"]["luts"] > small["resources"]["luts"]
        assert large["resources"]["ffs"] > small["resources"]["ffs"]

    def test_estimate_endpoint(self, client: TestClient) -> None:
        """The estimate endpoint returns an ECP5 estimate."""
        r = client.post(
            "/api/synth/estimate",
            json={"ir_op_count": 10, "target": "ecp5"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["target"] == "ecp5"
        assert data["estimated"] is True

    def test_estimate_endpoint_rejects_zero(self, client: TestClient) -> None:
        """An operation count of zero is rejected with 422."""
        r = client.post("/api/synth/estimate", json={"ir_op_count": 0, "target": "ice40"})
        assert r.status_code == 422

    def test_estimate_endpoint_rejects_non_integer(self, client: TestClient) -> None:
        """A non-integer operation count is rejected with 422."""
        r = client.post("/api/synth/estimate", json={"ir_op_count": "10", "target": "ice40"})
        assert r.status_code == 422
        assert "integer" in r.text

    def test_estimate_endpoint_rejects_unknown_target(self, client: TestClient) -> None:
        """An unknown target is rejected with 422."""
        r = client.post("/api/synth/estimate", json={"ir_op_count": 10, "target": "unknown"})
        assert r.status_code == 422
