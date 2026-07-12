# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — quantum-annealing modularisation benchmark tests

"""Exercise the parent/candidate benchmark as a real subprocess harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "benchmarks" / "bench_quantum_annealing_modularisation.py"
_RESULT = (
    _REPO_ROOT
    / "benchmarks"
    / "results"
    / "local_python_2026-07-12_quantum_annealing_modularisation.json"
)


def _current_source_digest() -> tuple[str, int]:
    """Hash the exact modular candidate source surface."""
    bridges = _REPO_ROOT / "src" / "sc_neurocore" / "bridges"
    files = [bridges / "quantum_annealing.py", *sorted(bridges.glob("annealing_*.py"))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), len(files)


def test_modularisation_benchmark_writes_path_free_comparison(tmp_path: Path) -> None:
    """Two real source probes produce raw samples, summaries, and deltas."""
    output = tmp_path / "quantum_annealing.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--baseline-root",
            str(_REPO_ROOT),
            "--candidate-root",
            str(_REPO_ROOT),
            "--baseline-label",
            "baseline",
            "--candidate-label",
            "candidate",
            "--iterations",
            "2",
            "--warmups",
            "1",
            "--no-affinity",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    raw = output.read_text(encoding="utf-8")
    assert str(_REPO_ROOT) not in raw
    assert "/home/" not in raw
    assert "/media/" not in raw
    payload: object = json.loads(raw)
    assert isinstance(payload, dict)
    assert payload["schema_version"] == (
        "sc-neurocore.quantum-annealing-modularisation-benchmark.v1"
    )
    variants = payload["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 2
    for variant in variants:
        assert isinstance(variant, dict)
        assert variant["local_path_recorded"] is False
        assert isinstance(variant["quantum_annealing_source_sha256"], str)
        assert variant["quantum_annealing_source_file_count"] == 10
    results = payload["results"]
    assert isinstance(results, dict)
    for label in ("baseline", "candidate"):
        result = results[label]
        assert isinstance(result, dict)
        assert result["sample_count"] == 2
        samples = result["samples"]
        assert isinstance(samples, list)
        for sample in samples:
            assert sample["import_ns"] > 0
            assert sample["compile_ns"] > 0
            assert sample["solve_ns"] > 0
    applicability = payload["polyglot_applicability"]
    assert isinstance(applicability, dict)
    assert applicability["python"] == "measured"
    assert applicability["rust"] == "maintained_native_kernel_benchmarked_separately"


def test_modularisation_benchmark_rejects_zero_iterations(tmp_path: Path) -> None:
    """Invalid sample counts fail before an evidence file is created."""
    output = tmp_path / "quantum_annealing.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--baseline-root",
            str(_REPO_ROOT),
            "--iterations",
            "0",
            "--output",
            str(output),
        ],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 2
    assert "iterations must be positive" in completed.stderr
    assert not output.exists()


def test_committed_modularisation_result_matches_candidate_source() -> None:
    """Committed evidence identifies the exact facade and nine modules."""
    payload: object = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    configuration = payload["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["iterations"] == 30
    assert configuration["warmups"] == 5
    variants = payload["variants"]
    assert isinstance(variants, list)
    candidate = next(
        variant
        for variant in variants
        if isinstance(variant, dict) and variant.get("label") == "quantum-modular"
    )
    assert isinstance(candidate, dict)
    digest, file_count = _current_source_digest()
    assert candidate["quantum_annealing_source_sha256"] == digest
    assert candidate["quantum_annealing_source_file_count"] == file_count == 10
    applicability = payload["polyglot_applicability"]
    assert isinstance(applicability, dict)
    assert applicability["go"] == "removed_empty_generated_mirror"
    assert applicability["julia"] == "removed_non_parsing_generated_mirror"
    assert applicability["mojo"] == "removed_non_parsing_generated_mirror"
