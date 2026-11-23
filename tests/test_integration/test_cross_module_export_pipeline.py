# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestExportPipeline from former test_cross_module.py

"""Focused suite: TestExportPipeline from former test_cross_module.py."""

from __future__ import annotations


class TestExportPipeline:
    """Verify Model Zoo → ONNX → TVM → MLIR → Verilog pipeline."""

    def test_pipeline_creation(self) -> None:
        """A new export pipeline has a model registry."""
        from sc_neurocore.export.pipeline import ExportPipeline

        p = ExportPipeline()
        assert p.registry is not None

    def test_pipeline_result_dataclass(self) -> None:
        """A result whose only stage succeeded is a success and names the stage."""
        from sc_neurocore.export.pipeline import PipelineResult, PipelineStageResult

        r = PipelineResult()
        r.stages.append(PipelineStageResult(stage="test", success=True, output="ok"))
        assert r.success
        assert "test" in r.summary()

    def test_pipeline_stage_failure(self) -> None:
        """A result with a failed stage is not a success."""
        from sc_neurocore.export.pipeline import PipelineResult, PipelineStageResult

        r = PipelineResult()
        r.stages.append(PipelineStageResult(stage="fail", success=False, output="err"))
        assert not r.success
