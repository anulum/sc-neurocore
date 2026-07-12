# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli hub tests

"""Exercise cli hub behaviour through the public CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.cli_test_support import run_cli


class TestHubInitCommand:
    """Tests for `sc-neurocore hub-init ...`."""

    def test_hub_init_writes_bundle(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        out = tmp_path / "hub"
        rc = run_cli(
            "hub-init",
            "--output",
            str(out),
            "--port",
            "8111",
            "--bind-host",
            "10.0.0.5",
            "--hub-image",
            "sc-neurocore:test",
            "--online",
        )

        assert rc == 0
        assert (out / "docker-compose.yml").exists()
        manifest = json.loads((out / "hub_manifest.json").read_text(encoding="utf-8"))
        assert manifest["services"]["studio"]["url"] == "http://10.0.0.5:8111"
        assert manifest["network_policy"]["ingress_scope"] == "private_network"
        assert manifest["network_policy"]["offline_environment"]["SC_NEUROCORE_HUB_OFFLINE"] == "0"
        assert "image: sc-neurocore:test" in (out / "docker-compose.yml").read_text(
            encoding="utf-8"
        )
        assert "hub bundle generated" in capsys.readouterr().out

    def test_hub_init_rejects_invalid_port(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cli("hub-init", "--output", str(tmp_path / "hub"), "--port", "0")

        assert rc == 1
        assert "studio_port must be in the range" in capsys.readouterr().out
