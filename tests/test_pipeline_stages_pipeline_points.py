# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipelinePoints from former test_pipeline_stages.py

"""Focused suite: TestPipelinePoints from former test_pipeline_stages.py."""

from __future__ import annotations

from tests.pipeline_stages_support import *  # noqa: F403


class TestPipelinePoints:
    """Verify user-specified pipeline insertion points."""

    def test_specific_point_only(self, izhikevich_neuron):
        """Only the specified multiply should be registered."""
        v = compile_to_verilog(
            izhikevich_neuron,
            pipeline_stages=0,
            pipeline_points=["_mul0"],
        )
        assert "_mul0_r" in v, "Expected _mul0 to be registered"
        # Other multiplies should NOT be registered when pipeline_stages=0
        # and they're not in pipeline_points

    def test_multiple_points(self, izhikevich_neuron):
        """Multiple specified points should all be registered."""
        v = compile_to_verilog(
            izhikevich_neuron,
            pipeline_stages=0,
            pipeline_points=["_mul0", "_mul1"],
        )
        assert "_mul0_r" in v
        assert "_mul1_r" in v
