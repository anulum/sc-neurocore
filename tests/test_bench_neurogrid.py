# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NeuroGrid Benchmark Runtime Contract Tests

from __future__ import annotations

from benchmarks import bench_model_neurogrid as benchmark


def test_go_benchmark_executes_without_backend_omission() -> None:
    """Require the real Go benchmark to yield every measured repetition."""
    result = benchmark.run_go_backend()

    assert result.get("skipped") is not True
    assert result["backend"] == "go"
    assert result["repeats"] == benchmark.REPEATS
    assert result["spikes"] == 94
    assert result["spike_counts"] == [94] * benchmark.REPEATS
