# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio synthesis pnr

"""Focused suite: TestPnREndpoint from former test_studio_synthesis.py."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from starlette.testclient import TestClient

from pathlib import Path
from tests.studio_synthesis_support import client as client


class TestPnREndpoint:
    """The Studio place-and-route endpoint and its input checks."""

    def test_pnr_requires_json_path(self, client: TestClient) -> None:
        """A place-and-route request without a netlist path is rejected with 422."""
        r = client.post("/api/synth/pnr", json={"target": "ice40"})
        assert r.status_code == 422

    def test_pnr_empty_path_rejected(self, client: TestClient) -> None:
        """An empty netlist path is rejected with 422."""
        r = client.post("/api/synth/pnr", json={"json_path": "", "target": "ice40"})
        assert r.status_code == 422

    def test_pnr_nonexistent_path(self, client: TestClient) -> None:
        """A missing netlist reports failure."""
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": "/tmp/does_not_exist.json", "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False

    def test_pnr_rejects_non_json_input(self, client: TestClient, tmp_path: Path) -> None:
        """A netlist that is not a JSON file is refused."""
        path = tmp_path / "design.txt"
        path.write_text("not a netlist", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert ".json netlist" in data["error"]

    def test_pnr_rejects_directory_input(self, client: TestClient, tmp_path: Path) -> None:
        """A directory is refused as a netlist."""
        directory = tmp_path / "dir_as_input.json"
        directory.mkdir()
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(directory), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "not a regular file" in data["error"]

    def test_pnr_rejects_non_json_payload(self, client: TestClient, tmp_path: Path) -> None:
        """A netlist that is not valid UTF-8 JSON is refused."""
        path = tmp_path / "invalid.json"
        path.write_text("this is not json", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "valid UTF-8 JSON" in data["error"]

    def test_pnr_rejects_non_object_json_payload(self, client: TestClient, tmp_path: Path) -> None:
        """A netlist whose JSON is not an object is refused."""
        path = tmp_path / "array_payload.json"
        path.write_text("[]", encoding="utf-8")
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(path), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "must be an object" in data["error"]

    def test_pnr_rejects_symlink_input(self, client: TestClient, tmp_path: Path) -> None:
        """A symlinked netlist is refused."""
        target = tmp_path / "netlist.json"
        target.write_text("{}", encoding="utf-8")
        symlink = tmp_path / "netlist_link.json"
        symlink.symlink_to(target)
        r = client.post(
            "/api/synth/pnr",
            json={"json_path": str(symlink), "target": "ice40"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["success"] is False
        assert "must not be a symlink" in data["error"]
