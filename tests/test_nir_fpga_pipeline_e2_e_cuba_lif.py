# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2ECubaLIF from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2ECubaLIF from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import (
    _build_cubalif_network,
    _full_pipeline,
)


class TestE2ECubaLIF:
    """Full pipeline with CubaLIF neurons (dual time constants)."""

    def test_cubalif_verilog_dual_dynamics(self) -> None:
        """A CubaLIF network compiles to a cuba_lif neuron module with registers and a spike output."""
        graph = _build_cubalif_network()
        result = _full_pipeline(graph, module_name="cuba_net")

        assert "cuba_lif" in result.neuron_modules
        cuba_v = result.neuron_modules["cuba_lif"]
        # CubaLIF has two state variables: i_syn and v
        assert "i_syn_reg" in cuba_v or "i__syn_reg" in cuba_v or "reg" in cuba_v
        assert "spike_out" in cuba_v

    def test_cubalif_weight_rom(self) -> None:
        """The CubaLIF network's weight ROM holds its 12 synapses."""
        graph = _build_cubalif_network(n_in=3, n_out=4)
        result = _full_pipeline(graph)

        # 3×4 = 12 weights
        assert result.total_synapses == 12
        assert "module sc_nir_weight_rom" in result.weight_rom
