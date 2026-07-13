# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for NIR bridge (import, node mapping, execution)

"""Tests for nir_bridge: NIR graph → SC-NeuroCore network conversion."""

import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.nir_bridge import SCNNetwork
from sc_neurocore.nir_bridge.parser import (
    SCNetwork,
)

from tests.test_nir_bridge.support import make_lif_affine_graph


def test_scnnetwork_alias_supports_from_nir_classmethod_and_hardware_compile() -> None:
    graph = make_lif_affine_graph(n_in=2, n_out=2)

    network = SCNNetwork.from_nir(graph, dt=1.0)
    result = network.to_hardware(
        module_name="api_lif_network",
        data_width=18,
        fraction=10,
        bitstream_length=512,
    )

    assert isinstance(network, SCNetwork)
    assert result.module_name == "api_lif_network"
    assert result.q_format == "Q8.10"
    assert result.total_neurons == 2
    assert "module api_lif_network" in result.top_module


def test_to_hardware_preserves_existing_compiler_validation() -> None:
    network = SCNetwork(nodes={}, edges=[], input_nodes=[], output_nodes=[])

    with pytest.raises(ValueError, match="at least one neuron population"):
        network.to_hardware(module_name="empty_network")
