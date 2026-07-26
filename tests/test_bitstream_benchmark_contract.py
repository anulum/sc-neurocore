# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Contract tests for the equal-operation bitstream benchmark methodology."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from numpy.typing import NDArray


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark() -> ModuleType:
    path = REPO_ROOT / "benchmarks" / "bench_bitstream_numpy_lut.py"
    spec = importlib.util.spec_from_file_location("bench_bitstream_numpy_lut", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numpy_lut_counts_actual_packed_bytes() -> None:
    benchmark = _load_benchmark()
    packed = np.array([0x00, 0x01, 0x55, 0xAA, 0xFF], dtype=np.uint8)
    assert benchmark.numpy_lut_popcount(packed) == 17


def test_patterned_oracle_handles_partial_and_large_cycles() -> None:
    benchmark = _load_benchmark()
    assert benchmark._patterned_expected_popcount(5) == 5
    assert benchmark._patterned_expected_popcount(256) == 1024
    assert benchmark._patterned_expected_popcount((1 << 30) // 8) == 1 << 29


@pytest.mark.parametrize(
    "invalid",
    [np.array([1], dtype=np.uint16), np.array([[1]], dtype=np.uint8)],
)
def test_numpy_lut_rejects_non_packed_byte_vectors(invalid: NDArray[np.generic]) -> None:
    benchmark = _load_benchmark()
    with pytest.raises(ValueError, match="one-dimensional uint8"):
        benchmark.numpy_lut_popcount(invalid)


def test_small_benchmark_receipt_is_source_bound() -> None:
    benchmark = _load_benchmark()
    payload = benchmark.run_benchmark((1024, 8192), warmups=0, repeats=2)

    assert payload["schema"] == "sc-neurocore.bitstream-popcount-benchmark.v1"
    assert payload["implementation"].startswith("NumPy uint8 lookup table")
    assert [result["bit_count"] for result in payload["results"]] == [1024, 8192]
    assert all(len(result["samples_ns"]) == 2 for result in payload["results"])
    assert set(payload["source_hashes"]) == {
        "benchmarks/bench_bitstream_numpy_lut.py",
        "engine/benches/bitstream_bench.rs",
    }


def test_rust_benchmark_compares_equal_operation_at_cache_stressing_sizes() -> None:
    source = (REPO_ROOT / "engine" / "benches" / "bitstream_bench.rs").read_text()
    assert "count_ones_baseline" in source
    assert 'BenchmarkId::new("u64_count_ones"' in source
    assert 'BenchmarkId::new("simd_dispatch"' in source
    assert "64 * MEBIBIT" in source
    assert "1024 * MEBIBIT" in source
    assert 'bench_function("pack_u8_bits_to_u64_1m"' in source


def test_historical_result_disclaims_old_pack_and_popcount_comparison() -> None:
    payload = json.loads(
        (REPO_ROOT / "benchmarks/results/criterion_bitstream_2026-03-26.json").read_text()
    )
    pack_result = next(result for result in payload["results"] if result["name"].startswith("pack"))
    assert "not a SIMD popcount path" in pack_result["note"]
    assert "not headline authority" in payload["evidence_class"]
    assert "equal-operation" in payload["comparison_to_readme_claim"]["conclusion"]


def test_benchmark_workflow_records_numpy_lut_artifact() -> None:
    workflow = (REPO_ROOT / ".github/workflows/benchmark.yml").read_text()
    assert "python benchmarks/bench_bitstream_numpy_lut.py" in workflow
    assert "benchmarks/results/ci_bitstream_numpy_lut.json" in workflow
    assert "path: benchmarks/results/" in workflow
