# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Amari field benchmark evidence gates

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from benchmarks import bench_amari_field as benchmark


def test_every_runtime_matches_complete_python_receipt() -> None:
    reference = benchmark._run_backend("python", 64)
    for backend in benchmark.BACKENDS[1:]:
        actual = benchmark._run_backend(backend, 64)
        np.testing.assert_allclose(
            actual["states"],
            reference["states"],
            rtol=0.0,
            atol=benchmark.PARITY_ATOL[backend],
        )
        np.testing.assert_array_equal(actual["mean_rates"], reference["mean_rates"])


def test_committed_evidence_is_source_bound_and_complete() -> None:
    payload = json.loads(benchmark.DEFAULT_OUTPUT.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["production_speed_claim"] is False
    assert payload["continuous_space_convergence_claimed"] is False
    assert payload["source_hashes"] == benchmark._source_hashes()
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True and row["used"] is True
        assert row["trace_matches_python"] is True
        assert row["mean_rates_exact"] is True


def test_live_native_binary_records_are_self_consistent() -> None:
    for record in benchmark._binary_hashes().values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert record["sha256"] == benchmark._sha256(path)
        assert record["size_bytes"] == path.stat().st_size


def test_rust_safety_evidence_executes() -> None:
    assert benchmark._verify_rust_safety()["passed"] is True
