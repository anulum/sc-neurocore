# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio structural graph verification HTTP contracts

"""Structural verification inspects a graph without claiming source replay."""

from __future__ import annotations

from starlette.testclient import TestClient

from tests.studio_compiler_support import LIF_IR_REQUEST, client as client


class TestIRVerify:
    """Exercise structural inspection of actual source-bound native graphs."""

    def test_verify_valid_ir(self, client: TestClient) -> None:
        """Report structural validity without replacing semantic replay."""
        build = client.post("/api/ir/build", json=LIF_IR_REQUEST).json()
        r = client.post("/api/ir/verify", json={"ir_text": build["ir_text"]})
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_verify_empty_ir_fails(self, client: TestClient) -> None:
        """Reject missing graph text before starting a worker."""
        r = client.post("/api/ir/verify", json={"ir_text": ""})
        assert r.status_code == 422
