# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCOnnxExporter performance contract

from __future__ import annotations

import time
from pathlib import Path

import pytest

from sc_neurocore.export.onnx_exporter import SCOnnxExporter
from tests.test_export.onnx_exporter_support import make_layers, perf_enabled
from tests.performance_guard import assert_load_tolerant_throughput


@pytest.mark.skipif(not perf_enabled(), reason="Set SC_NEUROCORE_PERF=1 to enable perf checks.")
def test_onnx_export_perf_small(tmp_path: Path) -> None:
    """Export a small model within the maintained sanity threshold."""

    path = tmp_path / "model.json"
    start = time.perf_counter()
    SCOnnxExporter.export(make_layers(), str(path))
    elapsed = time.perf_counter() - start
    assert_load_tolerant_throughput(
        label="ONNX export run", observed_per_second=1.0 / elapsed, strict_minimum_per_second=0.5
    )
