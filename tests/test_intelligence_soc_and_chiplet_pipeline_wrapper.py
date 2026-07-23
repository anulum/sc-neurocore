# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestPipelineWrapper from former test_intelligence_soc_and_chiplet.py

"""Focused suite: TestPipelineWrapper from former test_intelligence_soc_and_chiplet.py."""

from __future__ import annotations

from tests.intelligence_soc_and_chiplet_support import *  # noqa: F403

class TestPipelineWrapper:
    """Pipeline register insertion for high-frequency targets."""

    def test_basic_pipeline(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b + c"},
            data_width=16,
        )
        assert "module sc_lif_pipe" in v
        assert "endmodule" in v
        assert "valid_in" in v
        assert "valid_out" in v
        assert "latency" in v

    def test_pipeline_stages_in_output(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_hh",
            {"v": "a * b * c"},
            stages=3,
        )
        assert "I_pipe_0" in v
        assert "I_pipe_1" in v
        assert "I_pipe_2" in v
        assert "valid_pipe" in v

    def test_auto_stages_from_target(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper(
            "sc_lif",
            {"v": "a * b * c * d * e"},
            target="artix7",
        )
        assert "module sc_lif_pipe" in v
        assert "pipeline" in v.lower()

    def test_output_register(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_lif", {"v": "a * b"})
        assert "v_out" in v
        assert "spike_out" in v
        assert "v_reg" in v

    def test_inner_module_instantiation(self):
        from sc_neurocore.compiler.intelligence import (
            generate_pipeline_wrapper,
        )

        v = generate_pipeline_wrapper("sc_custom", {"v": "a + b"})
        assert "sc_custom core" in v
