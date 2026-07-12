# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli serve tests

"""Exercise cli serve behaviour through the public CLI."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from tests.cli_test_support import fake_module, run_cli


class TestServeCommand:
    """Tests for ``sc-neurocore serve`` input and server wiring."""

    def test_serve_without_model_arg_returns_1(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`sc-neurocore serve` with no model argument prints usage and exits 1."""
        rc = run_cli("serve")
        assert rc == 1
        out = capsys.readouterr().out
        assert "serve requires a model file" in out

    def test_serve_rejects_non_nir_extension(self, capsys: pytest.CaptureFixture[str]) -> None:
        """`.pt` and other extensions are not yet supported by serve; exit 1."""
        rc = run_cli("serve", "model.pt")
        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_serve_loads_nir_and_blocks_in_server(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Successful path: read NIR, build Network, start blocking SpikeServer."""
        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")  # contents are mocked away

        fake_graph = mock.MagicMock()
        fake_network = mock.MagicMock()
        fake_network.topo_order = ["a", "b", "c"]

        fake_server_instance = mock.MagicMock()
        fake_server_cls = mock.MagicMock(return_value=fake_server_instance)

        fake_nir = fake_module("nir", read=mock.MagicMock(return_value=fake_graph))
        fake_bridge = fake_module(
            "sc_neurocore.nir_bridge",
            from_nir=mock.MagicMock(return_value=fake_network),
        )
        fake_serve_mod = fake_module(
            "sc_neurocore.serve",
            SpikeServer=fake_server_cls,
        )

        with mock.patch.dict(
            "sys.modules",
            {
                "nir": fake_nir,
                "sc_neurocore.nir_bridge": fake_bridge,
                "sc_neurocore.serve": fake_serve_mod,
            },
        ):
            rc = run_cli("serve", str(nir_path), "--port", "8123", "--dt", "1.0")

        assert rc == 0
        fake_server_cls.assert_called_once_with(fake_network, port=8123)
        fake_server_instance.start.assert_called_once_with(blocking=True)
        assert "Loaded NIR graph with 3 nodes" in capsys.readouterr().out
