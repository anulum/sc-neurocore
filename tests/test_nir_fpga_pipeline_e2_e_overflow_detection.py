# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestE2EOverflowDetection from former test_nir_fpga_pipeline.py

"""Focused suite: TestE2EOverflowDetection from former test_nir_fpga_pipeline.py."""

from __future__ import annotations

from tests.nir_fpga_pipeline_support import *  # noqa: F403

class TestE2EOverflowDetection:
    """Extreme parameter values must produce warnings."""

    def test_large_tau_overflow_warning(self):
        """tau=50000 overflows Q8.8 max=127.996 — must warn."""
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff": nir.Affine(
                    weight=np.eye(2, dtype=np.float32),
                    bias=np.zeros(2, dtype=np.float32),
                ),
                "lif": nir.LIF(
                    tau=np.full(2, 50000.0),  # WAY out of range for Q8.8
                    r=np.ones(2),
                    v_leak=np.zeros(2),
                    v_threshold=np.ones(2),
                ),
                "output": nir.Output(output_type={"output": np.array([2])}),
            },
            edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
        )
        result = _full_pipeline(graph, data_width=16, fraction=8)

        # Must have overflow warnings
        assert len(result.warnings) > 0
        overflow_warns = [w for w in result.warnings if "Overflow" in w or "clamped" in w]
        assert len(overflow_warns) > 0, f"Expected overflow warnings, got: {result.warnings}"

    def test_large_weight_overflow(self):
        """Weights=500 overflows Q8.8 — must warn."""
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff": nir.Affine(
                    weight=np.full((2, 2), 500.0, dtype=np.float32),
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
        result = _full_pipeline(graph, data_width=16, fraction=8)

        overflow_warns = [w for w in result.warnings if "Overflow" in w]
        assert len(overflow_warns) > 0
