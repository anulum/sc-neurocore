# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — cli mapping tests

"""Exercise cli mapping behaviour through the public CLI."""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest import mock

import pytest

from tests.cli_test_support import fake_module, run_cli


class TestMapNirCommand:
    """Tests for `sc-neurocore map-nir ...`."""

    def test_map_nir_requires_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        """A missing model returns an actionable usage error."""
        assert run_cli("map-nir") == 1
        assert "requires a NIR model file" in capsys.readouterr().out

    def test_map_nir_rejects_non_nir_extension(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        rc = run_cli(
            "map-nir",
            str(tmp_path / "model.pt"),
            "--output",
            str(tmp_path / "out"),
            "--hardware-targets",
            "loihi2",
        )

        assert rc == 1
        assert "supports .nir files only" in capsys.readouterr().out

    def test_map_nir_writes_report_with_mocked_nir_import(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        nir_path = tmp_path / "model.nir"
        nir_path.write_bytes(b"")
        fake_graph = mock.MagicMock()
        fake_network = types.SimpleNamespace(
            nodes={
                "input": {"node_type": "Input", "shape": (2,)},
                "dense": {"node_type": "Linear", "weight": (3, 2)},
                "output": {"node_type": "Output", "shape": (3,)},
            },
            topo_order=["input", "dense", "output"],
            edges=[("input", "dense"), ("dense", "output")],
        )
        fake_nir = fake_module("nir", read=mock.MagicMock(return_value=fake_graph))

        with (
            mock.patch.dict("sys.modules", {"nir": fake_nir}),
            mock.patch("sc_neurocore.nir_bridge.from_nir", return_value=fake_network),
        ):
            rc = run_cli(
                "map-nir",
                str(nir_path),
                "--output",
                str(tmp_path / "mapping"),
                "--hardware-targets",
                "loihi2,spinnaker2,akida",
                "--dt",
                "0.5",
                "--T",
                "256",
            )

        report = json.loads(
            (tmp_path / "mapping" / "nir_silicon_mapping_report.json").read_text(encoding="utf-8")
        )
        assert rc == 0
        assert [target["target_id"] for target in report["targets"]] == [
            "loihi2",
            "spinnaker2",
            "akida",
        ]
        assert report["targets"][0]["summary"]["estimated_synapses"] == 6
        assert "NIR silicon mapping report generated" in capsys.readouterr().out

    def test_map_nir_rejects_empty_target_list(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        model = tmp_path / "model.nir"
        model.write_bytes(b"")
        rc = run_cli("map-nir", str(model), "--hardware-targets", ",")
        assert rc == 1
        assert "at least one target" in capsys.readouterr().out

    def test_map_nir_reports_import_failure(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """NIR importer failures cross the public boundary as status one."""
        model = tmp_path / "model.nir"
        model.write_bytes(b"")
        fake_nir = fake_module("nir", read=mock.MagicMock(side_effect=ValueError("bad graph")))

        with mock.patch.dict("sys.modules", {"nir": fake_nir}):
            assert run_cli("map-nir", str(model)) == 1

        assert "bad graph" in capsys.readouterr().out
