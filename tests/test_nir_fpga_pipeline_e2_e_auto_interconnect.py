# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2EAutoInterconnect from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2EAutoInterconnect from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2EAutoInterconnect:
    """Verify direct and weighted event interconnect selection."""

    def test_small_network_uses_direct(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        assert result.total_neurons == 10
        assert result.interconnect == "direct"
        assert "direct wiring" in result.top_module.lower() or "direct" in result.top_module.lower()
        # No AER bus signals
        assert "aer_addr" not in result.top_module

    def test_large_network_uses_weighted_event_interconnect(self):
        """Large networks use audited weighted event routing instead of warning-only direct fallback."""
        rng = np.random.RandomState(42)
        n_big = _AER_THRESHOLD + 10  # >64
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([4])}),
                "aff1": nir.Affine(
                    weight=rng.randn(n_big, 4).astype(np.float32),
                    bias=np.zeros(n_big, dtype=np.float32),
                ),
                "lif1": nir.LIF(
                    tau=np.full(n_big, 20.0),
                    r=np.ones(n_big),
                    v_leak=np.zeros(n_big),
                    v_threshold=np.ones(n_big),
                ),
                "output": nir.Output(output_type={"output": np.array([n_big])}),
            },
            edges=[("input", "aff1"), ("aff1", "lif1"), ("lif1", "output")],
        )
        result = _full_pipeline(graph)

        assert result.total_neurons > _AER_THRESHOLD
        assert result.interconnect == "aer"
        assert "weighted event routing" not in " ".join(result.warnings)
        assert "aer_addr" in result.top_module
        assert "aer_event_valid" in result.top_module
        assert f"output wire [{n_big - 1}:0] spike_bus" in result.top_module

    def test_large_two_layer_network_emits_weighted_event_fanout(self):
        """Spiking source populations must contribute signed event weights to destinations."""
        rng = np.random.RandomState(11)
        n_hidden = _AER_THRESHOLD
        n_out = 3
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff1": nir.Affine(
                    weight=rng.randn(n_hidden, 2).astype(np.float32),
                    bias=np.zeros(n_hidden, dtype=np.float32),
                ),
                "lif1": nir.LIF(
                    tau=np.full(n_hidden, 20.0),
                    r=np.ones(n_hidden),
                    v_leak=np.zeros(n_hidden),
                    v_threshold=np.ones(n_hidden),
                ),
                "aff2": nir.Affine(
                    weight=np.full((n_out, n_hidden), 0.5, dtype=np.float32),
                    bias=np.zeros(n_out, dtype=np.float32),
                ),
                "lif2": nir.LIF(
                    tau=np.full(n_out, 20.0),
                    r=np.ones(n_out),
                    v_leak=np.zeros(n_out),
                    v_threshold=np.ones(n_out),
                ),
                "output": nir.Output(output_type={"output": np.array([n_out])}),
            },
            edges=[
                ("input", "aff1"),
                ("aff1", "lif1"),
                ("lif1", "aff2"),
                ("aff2", "lif2"),
                ("lif2", "output"),
            ],
        )
        result = _full_pipeline(graph, module_name="weighted_event_net")

        assert result.interconnect == "aer"
        assert "weighted event fan-out accumulation" in result.top_module
        assert "if (p0_n0_spike)" in result.top_module
        assert "p1_n0_I_acc_next = p1_n0_I_acc_next + " in result.top_module
        assert "00080;" in result.top_module
