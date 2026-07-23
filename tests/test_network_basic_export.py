# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExport from former test_network_basic.py

"""Focused suite: TestExport from former test_network_basic.py."""

from __future__ import annotations

from tests.network_basic_support import *  # noqa: F403

class TestExport:
    def test_export_lif_network(self, tmp_path):
        pop = Population("LapicqueNeuron", 4)
        net = Network(pop)
        path = export_verilog(net, str(tmp_path / "verilog"))
        assert path.endswith(".v")
        with open(path) as f:
            content = f.read()
        assert "sc_lif_array" in content
        assert "sc_network_top" in content

    def test_export_rejects_unsupported(self, tmp_path):
        pop = Population("HodgkinHuxleyNeuron", 2)
        net = Network(pop)
        with pytest.raises(SCHardwareError, match="cannot be exported"):
            export_verilog(net, str(tmp_path / "verilog"))

    def test_export_creates_params_file(self, tmp_path):
        pop = Population("LapicqueNeuron", 8, label="layer0")
        net = Network(pop)
        export_verilog(net, str(tmp_path / "out"))
        params = (tmp_path / "out" / "params.vh").read_text()
        assert "POP_0_SIZE 8" in params
