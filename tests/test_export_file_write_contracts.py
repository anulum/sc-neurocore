# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for export file-write contracts

"""Contracts for propagating file-write failures from exporters."""

from __future__ import annotations

from unittest.mock import patch

import pytest


def test_onnx_exporter_propagates_write_errors() -> None:
    from sc_neurocore.export.onnx_exporter import SCOnnxExporter

    class Layer:
        n_inputs = 4

    with patch("builtins.open", side_effect=OSError("test")), pytest.raises(OSError):
        SCOnnxExporter.export([Layer()], "model.json")


def test_verilog_generator_propagates_write_errors() -> None:
    from sc_neurocore.hdl_gen.verilog_generator import VerilogGenerator

    generator = VerilogGenerator(module_name="test_mod")

    with patch("builtins.open", side_effect=OSError("test")), pytest.raises(OSError):
        generator.save_to_file("output.v")
