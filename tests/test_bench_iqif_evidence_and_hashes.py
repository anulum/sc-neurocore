# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (evidence_and_hashes) from former test_bench_iqif.py

from __future__ import annotations

from tests.bench_iqif_support import *  # noqa: F403


def test_real_benchmark_writes_five_backend_integer_parity_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise all five public dispatchers with the pinned source prefix."""
    monkeypatch.setattr(benchmark, "N_STEPS", 15)
    monkeypatch.setattr(benchmark, "N_REPEATS", 1)
    monkeypatch.setattr(benchmark, "WARMUP_STEPS", 1)
    monkeypatch.setattr(benchmark, "_verify_rust_safety", _passing_safety)
    output = tmp_path / "iqif.json"

    assert benchmark.main(["--json", str(output), "--allow-unpinned"]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["workload"]["n_steps"] == 15
    assert payload["workload"]["current"] == 10
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    assert payload["auto_backend_order"][-1] == "python"
    reference_hash = payload["backends"]["python"]["trace_sha256"]
    assert all(row["trace_matches_python"] for row in payload["backends"].values())
    assert all(row["event_count_matches_python"] for row in payload["backends"].values())
    assert all(row["final_state_matches_python"] for row in payload["backends"].values())
    assert all(row["trace_sha256"] == reference_hash for row in payload["backends"].values())
    assert payload["backends"]["python"]["event_count"] == 1
    assert payload["backends"]["python"]["final_state"] == {"v": 128}


def test_source_hashes_cover_declared_implementation_surfaces() -> None:
    """Pin every declared source digest to the live file bytes."""
    hashes = benchmark._source_hashes()
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected
        assert len(expected) == 64


def test_binary_hashes_bind_loaded_native_artifacts() -> None:
    """Pin every measured native lane to the exact loaded file bytes."""
    records = benchmark._binary_hashes()
    assert set(records) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for record in records.values():
        path = Path(str(record["path"]))
        if not path.is_absolute():
            path = benchmark.REPOSITORY / path
        assert path.is_file()
        assert record["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert record["size_bytes"] == path.stat().st_size


def test_committed_evidence_matches_live_sources_and_full_parity() -> None:
    """Bind the measured five-lane artefact to current code and behaviour."""
    artifact = benchmark.REPOSITORY / "benchmarks/results/local_python_2026-07-14_iqif.json"
    artifact_text = artifact.read_text(encoding="utf-8")
    assert str(benchmark.REPOSITORY) not in artifact_text
    payload = json.loads(artifact_text)

    assert payload["schema_version"] == "sc-neurocore.polyglot-benchmark.v1"
    assert payload["kernel"] == benchmark.KERNEL
    assert payload["evidence_class"] == "local_regression_single_cpu_affinity_non_exclusive"
    assert payload["production_speed_claim"] is False
    assert payload["hardware_measurement_claimed"] is False
    assert payload["workload"] == {
        "current": benchmark.CURRENT,
        "initial_state": {
            "a": 1,
            "b": 1,
            "v": 128,
            "v_max": 255,
            "v_min": 0,
            "v_reset": 128,
            "v_rest": 128,
            "v_threshold": 200,
        },
        "n_steps": benchmark.N_STEPS,
        "parameters": "pinned Wu et al. tutorial defaults; complete signed-int32 ABI",
        "repeats": benchmark.N_REPEATS,
        "warmup_steps": benchmark.WARMUP_STEPS,
    }
    assert set(payload["measured_order"]) == set(benchmark.BACKENDS)
    native_order = [name for name in payload["measured_order"] if name != "python"]
    assert payload["fastest_measured_native_backend"] == native_order[0]
    assert payload["auto_backend_order"] == [*native_order, "python"]
    assert payload["recommended_auto_backend"] == native_order[0]
    assert payload["auto_backend_selection_basis"] == (
        "same-host measured warm batch order; non-exclusive timings remain diagnostic"
    )
    assert payload["verification"]["rust_safety"]["passed"] is True
    assert payload["meta"]["single_cpu_pinned"] is True
    assert payload["meta"]["exclusive_cpu_isolation_claimed"] is False

    reference = payload["backends"]["python"]
    assert reference["event_count"] == 13_333
    assert reference["final_state"] == {"v": 165}
    for backend in benchmark.BACKENDS:
        row = payload["backends"][backend]
        assert row["available"] is True
        assert row["used"] is True
        assert row["trace_sha256"] == reference["trace_sha256"]
        assert row["trace_matches_python"] is True
        assert row["trace_mismatch_count"] == 0
        assert row["event_count"] == reference["event_count"]
        assert row["event_count_matches_python"] is True
        assert row["final_state"] == reference["final_state"]
        assert row["final_state_matches_python"] is True
        assert row["parity_max_abs_diff"] == 0

    hashes = payload["source_hashes"]
    for relative in benchmark.SOURCE_PATHS:
        expected = hashlib.sha256((benchmark.REPOSITORY / relative).read_bytes()).hexdigest()
        assert hashes[relative] == expected

    # Native artifacts are rebuilt per environment; their exact bytes are a
    # reproducible-build property, not part of this local-regression fidelity
    # claim (evidence_class is local_regression, production_speed_claim is
    # False). Bind the committed provenance shape and confirm the current tree
    # still produces the three loadable native lanes, rather than demanding
    # cross-build byte-for-byte reproducibility (see
    # test_binary_hashes_bind_loaded_native_artifacts for live self-consistency).
    committed_binaries = payload["binary_hashes"]
    live_binaries = benchmark._binary_hashes()
    assert set(committed_binaries) == set(live_binaries)
    assert set(live_binaries) == {"rust_extension", "go_shared_library", "mojo_shared_library"}
    for name in live_binaries:
        recorded = committed_binaries[name]
        assert len(str(recorded["sha256"])) == 64
        assert isinstance(recorded["size_bytes"], int)
        assert recorded["size_bytes"] > 0
        assert isinstance(recorded["path"], str)
        assert recorded["path"]
