# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExport from former test_network_basic.py

"""Focused suite: TestExport from former test_network_basic.py."""

from __future__ import annotations

from pathlib import Path

from tests.network_basic_support import *  # noqa: F403


class TestExport:
    """Files, names and refusals of the network Verilog exporter."""

    def test_export_lif_network(self, tmp_path: Path) -> None:
        """An SC LIF population exports its compiled neuron module and the top."""
        pop = Population("SCLapicqueLIFNeuron", 4)
        net = Network(pop)
        path = export_verilog(net, str(tmp_path / "verilog"))
        assert path.endswith(".v")
        with open(path) as f:
            content = f.read()
        assert "module sc_net_pop0_neuron" in content
        assert "sc_network_top" in content

    def test_export_rejects_unsupported(self, tmp_path: Path) -> None:
        """A model without an exported profile is refused before anything is written."""
        pop = Population("HodgkinHuxleyNeuron", 2)
        net = Network(pop)
        with pytest.raises(SCHardwareError, match="cannot be exported"):
            export_verilog(net, str(tmp_path / "verilog"))
        assert not (tmp_path / "verilog").exists()

    def test_export_creates_params_file(self, tmp_path: Path) -> None:
        """params.vh records each population's size."""
        pop = Population("SCLapicqueLIFNeuron", 8, label="layer0")
        net = Network(pop)
        export_verilog(net, str(tmp_path / "out"))
        params = (tmp_path / "out" / "params.vh").read_text()
        assert "POP_0_SIZE 8" in params
