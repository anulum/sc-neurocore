# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for SC-NIR FPGA integration

"""SC-NIR metadata integration tests for FPGA compilation artefacts."""

from __future__ import annotations

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.ir import scnir_to_dict, validate_scnir_dict
from sc_neurocore.nir_bridge import compile_network_to_fpga, from_nir, from_scnetwork


def _graph() -> object:
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([2])}),
            "aff": nir.Affine(
                weight=np.array([[0.25, -0.5], [0.75, 0.125]], dtype=np.float32),
                bias=np.zeros(2, dtype=np.float32),
            ),
            "lif": nir.LIF(
                tau=np.full(2, 20.0),
                r=np.ones(2),
                v_leak=np.zeros(2),
                v_threshold=np.ones(2),
            ),
            "output": nir.Output(output_type={"output": np.array([2])}),
        },
        edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
    )


def test_fpga_compile_result_carries_valid_scnir_metadata() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    result = compile_network_to_fpga(
        neuron_graph,
        module_name="scnir_integrated",
        data_width=18,
        fraction=10,
        bitstream_length=2048,
    )

    payload = scnir_to_dict(result.scnir_document)
    validate_scnir_dict(payload)
    assert {stream["bitstream_length"] for stream in payload["streams"]} == {2048}
    assert {stream["precision"]["total_bits"] for stream in payload["streams"]} == {18}
    assert {stream["precision"]["fractional_bits"] for stream in payload["streams"]} == {10}
    assert "localparam integer SCNIR_BITSTREAM_LENGTH = 2048;" in result.top_module
    assert "localparam integer SCNIR_STREAM_COUNT = 2;" in result.top_module


def test_fpga_compile_rejects_invalid_scnir_bitstream_length() -> None:
    network = from_nir(_graph(), dt=1.0)
    neuron_graph = from_scnetwork(network, dt=1.0)

    with pytest.raises(ValueError, match="bitstream_length"):
        compile_network_to_fpga(neuron_graph, bitstream_length=0)
