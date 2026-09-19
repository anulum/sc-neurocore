# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Evidence contract for the controlled SC stochastic adaptation benchmark."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks import bench_model_sc_stochastic_rate_adaptation as benchmark
from sc_neurocore.accel.sc_stochastic_rate_adaptation import PARITY_ATOL


def test_committed_record_is_source_bound_and_has_five_parity_lanes() -> None:
    """Reject stale source hashes or missing runtime evidence."""
    record = json.loads(
        (benchmark.ROOT / "benchmarks/results/bench_sc_stochastic_rate_adaptation.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["source_hashes"] == benchmark._hashes()
    assert record["steps"] == benchmark.STEPS
    assert record["repeats"] == benchmark.REPEATS
    assert set(record["backend_summary"]) == set(benchmark.BACKENDS)
    events = {row["events"] for row in record["backend_summary"].values()}
    assert len(events) == 1
    assert len(record["event_trace_sha256"]) == 64
    for backend, row in record["backend_summary"].items():
        assert row["median_ns"] > 0
        assert row["max_adaptation_error"] <= PARITY_ATOL[backend]


def test_real_short_benchmark_exercises_every_public_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Generate a new five-runtime record through the public dispatch."""
    monkeypatch.setattr(benchmark, "STEPS", 8)
    monkeypatch.setattr(benchmark, "REPEATS", 1)
    output = tmp_path / "sra.json"
    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    record = json.loads(output.read_text(encoding="utf-8"))
    assert set(record["backend_summary"]) == set(benchmark.BACKENDS)
    assert len({row["events"] for row in record["backend_summary"].values()}) == 1


def test_unpinned_and_missing_backend_fail_before_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep partial or uncontrolled measurements out of the evidence file."""
    output = tmp_path / "invalid.json"
    monkeypatch.setattr(os, "sched_getaffinity", lambda _pid: {0, 1})
    assert benchmark.main(["--json", str(output)]) == 2
    monkeypatch.setattr(benchmark, "backend_available", lambda backend: backend == "python")
    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 2
    assert not output.exists()


def test_source_files_match_current_bytes() -> None:
    """Ensure the generator hashes the intended model and benchmark files."""
    hashes = benchmark._hashes()
    for key, relative in benchmark.SOURCE_FILES.items():
        expected = hashlib.sha256((benchmark.ROOT / relative).read_bytes()).hexdigest()
        assert hashes[key] == expected
