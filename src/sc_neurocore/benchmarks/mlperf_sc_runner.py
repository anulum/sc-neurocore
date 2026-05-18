# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — MLPerf-SC fixture runner

"""Low-load deterministic MLPerf-SC fixture runner."""

from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

from .mlperf_sc_schema import (
    MLPERF_SC_RESULT_SCHEMA_VERSION,
    MLPerfSCValidationError,
    validate_mlperf_sc_result,
)

_FIXTURE_MODELS = frozenset({"fixture_sc_linear", "fixture_external_majority"})


def run_mlperf_sc_fixture(
    *,
    output_dir: str | Path,
    task: str = "synthetic_sc_xor",
    model: str = "fixture_sc_linear",
    seed: int = 0,
    bitstream_length: int = 256,
) -> Path:
    """Run the deterministic synthetic MLPerf-SC fixture and return result path."""

    if bitstream_length <= 0:
        raise MLPerfSCValidationError("execution.bitstream_length must be positive")
    if seed < 0:
        raise MLPerfSCValidationError("execution.seed must be non-negative")
    if task != "synthetic_sc_xor":
        raise MLPerfSCValidationError("fixture runner currently supports synthetic_sc_xor")
    if model not in _FIXTURE_MODELS:
        raise MLPerfSCValidationError(
            "fixture runner currently supports fixture_sc_linear and "
            "fixture_external_majority"
        )

    output = Path(output_dir)
    artifacts_dir = output / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_payload = _synthetic_xor_raw_payload(
        seed=seed,
        bitstream_length=bitstream_length,
        model=model,
    )
    raw_path = artifacts_dir / "synthetic_sc_xor_raw_results.json"
    _write_canonical_json(raw_path, raw_payload)

    environment_payload = _environment_payload()
    environment_path = artifacts_dir / "environment_manifest.json"
    _write_canonical_json(environment_path, environment_payload)

    result_payload: dict[str, Any] = {
        "schema_version": MLPERF_SC_RESULT_SCHEMA_VERSION,
        "run": {
            "run_id": f"{task}-{model}-seed{seed}-l{bitstream_length}",
            "task": task,
            "model": model,
            "dataset": "synthetic_xor",
            "started_at": "1970-01-01T00:00:00+00:00",
            "producer": raw_payload["producer"],
        },
        "execution": {
            "backend": "python",
            "target": "cpu",
            "sc_mode": raw_payload["sc_mode"],
            "bitstream_length": bitstream_length,
            "seed": seed,
        },
        "metrics": {
            "accuracy": raw_payload["accuracy"],
            "latency_ms": raw_payload["latency_ms"],
            "throughput_inferences_per_s": raw_payload["throughput_inferences_per_s"],
            "energy_j_per_inference": None,
            "power_w": None,
            "area": {
                "luts": None,
                "ffs": None,
                "bram": None,
                "dsp": None,
            },
        },
        "evidence": {
            "evidence_class": "simulation",
            "environment": {
                "python": environment_payload["python"],
                "platform": environment_payload["platform"],
            },
            "artifacts": [
                {
                    "kind": "raw_results",
                    "path": raw_path.relative_to(output).as_posix(),
                    "sha256": _sha256_file(raw_path),
                },
                {
                    "kind": "environment_manifest",
                    "path": environment_path.relative_to(output).as_posix(),
                    "sha256": _sha256_file(environment_path),
                },
            ],
        },
    }
    validate_mlperf_sc_result(result_payload, artifact_root=output)
    result_path = output / "mlperf_sc_result.json"
    _write_canonical_json(result_path, result_payload)
    return result_path


def _synthetic_xor_raw_payload(
    *,
    seed: int,
    bitstream_length: int,
    model: str,
) -> dict[str, Any]:
    baseline = _fixture_baseline(model)
    samples = []
    for inputs, target in [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 0),
    ]:
        samples.append(
            {
                "input": inputs,
                "target": target,
                "prediction": baseline["predictions"][tuple(inputs)],
            }
        )
    correct = sum(sample["target"] == sample["prediction"] for sample in samples)
    latency_ms = round((bitstream_length * len(samples)) / 1_000_000.0, 9)
    return {
        "task": "synthetic_sc_xor",
        "model": model,
        "producer": baseline["producer"],
        "baseline_family": baseline["baseline_family"],
        "evidence_boundary": baseline["evidence_boundary"],
        "sc_mode": baseline["sc_mode"],
        "seed": seed,
        "bitstream_length": bitstream_length,
        "samples": samples,
        "correct": correct,
        "total": len(samples),
        "accuracy": correct / len(samples),
        "latency_ms": latency_ms,
        "throughput_inferences_per_s": round(1000.0 / max(latency_ms, 1e-12), 6),
    }


def _fixture_baseline(model: str) -> dict[str, Any]:
    if model == "fixture_sc_linear":
        return {
            "producer": "sc-neurocore",
            "baseline_family": "sc_neurocore_fixture",
            "evidence_boundary": "deterministic fixture, not measured hardware",
            "sc_mode": "bipolar",
            "predictions": {
                (0, 0): 0,
                (0, 1): 1,
                (1, 0): 1,
                (1, 1): 0,
            },
        }
    if model == "fixture_external_majority":
        return {
            "producer": "external-reference-fixture",
            "baseline_family": "external_reference",
            "evidence_boundary": "deterministic fixture, not measured hardware",
            "sc_mode": "deterministic_replay",
            "predictions": {
                (0, 0): 0,
                (0, 1): 0,
                (1, 0): 0,
                (1, 1): 0,
            },
        }
    raise MLPerfSCValidationError(
        "fixture runner currently supports fixture_sc_linear and fixture_external_majority"
    )


def _environment_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "platform": f"{sys.platform}-{platform.machine()}",
    }


def _write_canonical_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
