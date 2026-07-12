# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DNA mapper import benchmark tests

"""Exercise the DNA mapper import benchmark as a real subprocess harness."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "benchmarks" / "bench_dna_mapper_import.py"
_RESULT = _REPO_ROOT / "benchmarks" / "results" / "local_python_2026-07-12_dna_mapper_import.json"


def _current_source_digest() -> str:
    """Hash the modular candidate exactly as the benchmark harness does."""
    digest = hashlib.sha256()
    bridges = _REPO_ROOT / "src" / "sc_neurocore" / "bridges"
    files = [bridges / "__init__.py", *sorted(bridges.glob("dna_*.py"))]
    for path in files:
        digest.update(path.relative_to(_REPO_ROOT).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_dna_import_benchmark_writes_path_free_comparison(tmp_path: Path) -> None:
    """Two real probes produce raw samples, summaries, and deltas."""
    output = tmp_path / "dna_mapper_import.json"
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
    payload: object = json.loads(raw)
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "sc-neurocore.dna-mapper-import-benchmark.v1"
    variants = payload["variants"]
    assert isinstance(variants, list)
    assert len(variants) == 2
    for variant in variants:
        assert isinstance(variant, dict)
        assert variant["local_path_recorded"] is False
        assert isinstance(variant["dna_mapper_source_sha256"], str)
    results = payload["results"]
    assert isinstance(results, dict)
    for label in ("baseline", "candidate"):
        result = results[label]
        assert isinstance(result, dict)
        assert result["sample_count"] == 2
        samples = result["samples"]
        assert isinstance(samples, list)
        assert all(sample["import_ns"] > 0 for sample in samples)
        assert all(sample["operation_ns"] > 0 for sample in samples)
    applicability = payload["polyglot_applicability"]
    assert isinstance(applicability, dict)
    assert applicability["python"] == "measured"
    assert applicability["rust"] == "verified_unmodified_6_tests"


def test_dna_import_benchmark_rejects_zero_iterations(tmp_path: Path) -> None:
    """Invalid sample counts fail before creating an evidence artefact."""
    output = tmp_path / "dna_mapper_import.json"
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


def test_committed_dna_import_result_matches_candidate_source() -> None:
    """Committed evidence identifies the exact modular source surface."""
    payload: object = json.loads(_RESULT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert payload["schema_version"] == "sc-neurocore.dna-mapper-import-benchmark.v1"
    configuration = payload["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["iterations"] == 30
    assert configuration["warmups"] == 5
    assert configuration["operations_per_sample"] == 10
    variants = payload["variants"]
    assert isinstance(variants, list)
    candidate = next(
        variant
        for variant in variants
        if isinstance(variant, dict) and variant.get("label") == "dna-modular"
    )
    assert isinstance(candidate, dict)
    assert candidate["dna_mapper_source_sha256"] == _current_source_digest()
    assert payload["polyglot_applicability"] == {
        "go": "not_applicable_no_runtime_mirror",
        "julia": "not_applicable_no_runtime_mirror",
        "mojo": "not_applicable_no_runtime_mirror",
        "python": "measured",
        "reason": (
            "This benchmark measures Python import and orchestration; the maintained Rust "
            "safety mirror is verified separately, while false generated Julia, Mojo, and "
            "Go scaffolds were removed"
        ),
        "rust": "verified_unmodified_6_tests",
    }
