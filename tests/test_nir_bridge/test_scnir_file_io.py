# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR conversion wiring

"""SC-NIR conversion tests for the NIR/NeuronGraph pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.cli import main
from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.ir.scnir_convert import (
    SCNIRConversionConfig,
    export_scnir_from_nir,
)

from tests.test_nir_bridge.scnir_delay_graphs import (
    _build_small_lif_graph,
)


def test_scnir_export_from_nir_file_round_trips(tmp_path: Path) -> None:
    model_path = tmp_path / "model.nir"
    output_path = tmp_path / "model.scnir.json"
    nir.write(str(model_path), _build_small_lif_graph())

    document = export_scnir_from_nir(
        model_path,
        output_path=output_path,
        config=SCNIRConversionConfig(bitstream_length=1024, base_seed=19),
    )

    assert json.loads(output_path.read_text(encoding="utf-8")) == scnir_to_dict(document)


def test_scnir_export_cli_writes_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    model_path = tmp_path / "model.nir"
    output_path = tmp_path / "export.scnir.json"
    nir.write(str(model_path), _build_small_lif_graph())

    with mock.patch(
        "sys.argv",
        [
            "sc-neurocore",
            "scnir",
            "export",
            str(model_path),
            "--output",
            str(output_path),
            "--T",
            "1024",
        ],
    ):
        rc = main()

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    validate_scnir_dict(payload)
    assert "SC-NIR exported" in capsys.readouterr().out
