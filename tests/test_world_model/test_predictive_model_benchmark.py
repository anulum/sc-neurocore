# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Predictive-model benchmark evidence contracts

"""Validate committed LGSSM evidence and the real reduced benchmark CLI."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import cast

from sc_neurocore.world_model import _lgssm_backends as backends


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "benchmarks/bench_predictive_model.py"
ARTIFACT = ROOT / "benchmarks/results/bench_predictive_model.json"
EXPECTED_BACKENDS = {"python", "rust", "julia", "go", "mojo"}
EXPECTED_SOURCES = {
    "src/sc_neurocore/world_model/predictive_model.py",
    "src/sc_neurocore/world_model/_lgssm_types.py",
    "src/sc_neurocore/world_model/_lgssm_backends.py",
    "src/sc_neurocore/world_model/_lgssm_filter.py",
    "src/sc_neurocore/world_model/_lgssm_smoothing.py",
    "src/sc_neurocore/world_model/_lgssm_em.py",
    "src/sc_neurocore/world_model/_predictive_world_model.py",
    "engine/src/lgssm.rs",
    "engine/src/lib.rs",
    "bridge/sc_neurocore_engine/__init__.py",
    "bridge/sc_neurocore_engine/world_model.py",
    "src/sc_neurocore/accel/rust/safety/predictive_model.rs",
    "src/sc_neurocore/accel/julia/world_model/predictive_model.jl",
    "src/sc_neurocore/accel/go/lgssm/lgssm.go",
    "src/sc_neurocore/accel/mojo/world_model/lgssm.mojo",
    "benchmarks/bench_predictive_model.py",
}


def _payload(path: Path = ARTIFACT) -> dict[str, object]:
    """Load a benchmark payload as a typed JSON mapping."""
    return cast(dict[str, object], json.loads(path.read_text(encoding="utf-8")))


def _mapping(value: object) -> dict[str, object]:
    """Narrow a known JSON mapping at a schema boundary."""
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _sha256(path: Path) -> str:
    """Return a file digest for provenance assertions."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_committed_evidence_discloses_loaded_host_scope() -> None:
    payload = _payload()
    isolation = _mapping(payload["isolation"])
    host = _mapping(payload["host"])
    git = _mapping(payload["git"])

    assert payload["schema_version"] == 2
    assert payload["evidence_class"] == "local_regression"
    assert isolation["classification"] == "loaded_host"
    assert isolation["exclusive_core_reserved"] is False
    assert isolation["other_heavy_jobs_running"] in {"yes", "no"}
    assert isolation["other_heavy_jobs_note"] != "not disclosed"
    assert host["exclusive_core_reserved"] is False
    assert isinstance(host["process_affinity"], list)
    assert len(cast(list[object], host["load_average_before"])) == 3
    assert len(cast(list[object], host["load_average_after"])) == 3
    assert git["source_binding"] == "sha256_per_file"
    assert payload["working_directory"] == "repository_root"
    assert "benchmarks/bench_predictive_model.py" in cast(str, payload["command"])
    assert not cast(str, payload["command"]).startswith("/")
    assert "not promotion-grade" in cast(str, payload["interpretation"])


def test_committed_evidence_covers_all_forward_backends_with_parity() -> None:
    payload = _payload()
    availability = _mapping(payload["backend_availability"])
    dispatch_policy = _mapping(payload["dispatch_policy"])
    forward = _mapping(payload["forward_filter"])
    workload = _mapping(payload["workload"])
    tolerances = _mapping(payload["parity_tolerances"])

    assert set(cast(list[str], payload["requested_backends"])) == EXPECTED_BACKENDS
    assert set(availability) == EXPECTED_BACKENDS
    assert set(forward) == EXPECTED_BACKENDS
    assert payload["dispatch_order"] == list(backends.AUTO_BACKEND_ORDER)
    measured_order = sorted(
        EXPECTED_BACKENDS,
        key=lambda backend: cast(float, _mapping(forward[backend])["median_ms"]),
    )
    assert payload["measured_order"] == measured_order
    assert dispatch_policy == {
        "basis": "post_import_probe_cost_then_interleaved_median_ms",
        "material_inversion_ratio": 1.1,
        "tie_policy": "preserve_stable_order",
        "warm_order_exceptions": [
            "rust_before_julia_avoids_lazy_julia_initialisation",
        ],
    }
    dispatch_order = cast(list[str], payload["dispatch_order"])
    materiality_ratio = cast(float, dispatch_policy["material_inversion_ratio"])
    for earlier_index, earlier_backend in enumerate(dispatch_order):
        earlier_median = cast(float, _mapping(forward[earlier_backend])["median_ms"])
        for later_backend in dispatch_order[earlier_index + 1 :]:
            if (earlier_backend, later_backend) == ("rust", "julia"):
                continue
            later_median = cast(float, _mapping(forward[later_backend])["median_ms"])
            assert earlier_median <= later_median * materiality_ratio
    assert workload["control_dim"] == 2
    assert workload["forward_sampling"] == "round_robin_rotating_start"
    repeats = cast(int, workload["repeats"])
    assert repeats % len(EXPECTED_BACKENDS) == 0

    for backend in EXPECTED_BACKENDS:
        available = _mapping(availability[backend])
        row = _mapping(forward[backend])
        parity = _mapping(row["parity_vs_python"])
        assert available["available"] is True
        assert available["reason"] == ""
        assert cast(float, available["post_import_probe_ms"]) >= 0.0
        assert len(cast(list[float], row["samples_ms"])) == repeats
        for field in (
            "means_max_abs",
            "covariances_max_abs",
            "pred_means_max_abs",
            "pred_covariances_max_abs",
        ):
            assert cast(float, parity[field]) <= cast(float, tolerances["array_max_abs"])
        assert cast(float, parity["log_likelihood_abs"]) <= cast(
            float,
            tolerances["log_likelihood_abs"],
        )


def test_committed_evidence_binds_sources_and_native_binaries() -> None:
    payload = _payload()
    source_hashes = _mapping(payload["source_sha256"])
    binary_evidence = _mapping(payload["binary_evidence"])

    assert set(source_hashes) == EXPECTED_SOURCES
    for relative_path, recorded_digest in source_hashes.items():
        assert recorded_digest == _sha256(ROOT / relative_path)

    assert set(binary_evidence) == {"rust", "go", "mojo"}
    for backend, raw_evidence in binary_evidence.items():
        evidence = _mapping(raw_evidence)
        digest = cast(str, evidence["sha256"])
        assert len(digest) == 64
        assert cast(int, evidence["size_bytes"]) > 0
        recorded_path = Path(cast(str, evidence["path"]))
        runtime_path = recorded_path if recorded_path.is_absolute() else ROOT / recorded_path
        # The recorded sha256 is provenance of the generating build host, not a
        # cross-host byte pin: Mojo/native linkers embed host-specific data, so the .so
        # bytes are reproducible only within one host and differ across the heterogeneous
        # CI build fleet (verified: identical Mojo 0.26.2.0 + x86-64-v3 yields distinct
        # .so on different machines). Bind the live backend as built and non-empty rather
        # than byte-pinning a foreign host's output, matching the sibling population
        # benchmarks' relaxed binary contract.
        if runtime_path.is_file():
            assert runtime_path.stat().st_size > 0


def test_python_only_workloads_are_explicit_not_native_skips() -> None:
    python_only = _mapping(_payload()["python_only_workloads"])
    assert set(python_only) == {"rts_smoother", "em_learner"}
    for workload in python_only.values():
        assert _mapping(workload)["implementation_scope"] == "python_only"


def test_reduced_python_cli_writes_valid_real_evidence(tmp_path: Path) -> None:
    output_path = tmp_path / "predictive_model_benchmark.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--backends",
            "python",
            "--steps",
            "8",
            "--repeats",
            "1",
            "--em-iterations",
            "1",
            "--other-heavy-jobs-running",
            "yes",
            "--other-heavy-jobs-note",
            "focused benchmark contract test",
            "--isolation-note",
            "ordinary test process; no exclusive core",
            "--json",
            str(output_path),
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    payload = _payload(output_path)
    assert payload["requested_backends"] == ["python"]
    assert set(_mapping(payload["forward_filter"])) == {"python"}
    assert _mapping(payload["binary_evidence"]) == {}
    assert _mapping(payload["workload"])["time_steps"] == 8
