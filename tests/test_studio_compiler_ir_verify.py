# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio compiler ir verify

"""Focused suite: TestIRVerify from former test_studio_compiler.py."""

from __future__ import annotations

from tests.studio_compiler_support import *  # noqa: F403


class TestIRVerify:
    def test_verify_valid_ir(self, client):
        build = client.post("/api/ir/build", json=LIF_EQ).json()
        r = client.post("/api/ir/verify", json={"ir_text": build["ir_text"]})
        assert r.status_code == 200
        data = r.json()
        assert data["valid"] is True
        assert data["errors"] == []

    def test_verify_empty_ir_fails(self, client):
        r = client.post("/api/ir/verify", json={"ir_text": ""})
        assert r.status_code == 422
