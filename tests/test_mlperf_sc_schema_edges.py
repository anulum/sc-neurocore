# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MLPerf-SC result schema validation edges

"""Contracts for MLPerf-SC result schema validation, evidence rules and helper guards."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sc_neurocore.benchmarks import (
    MLPERF_SC_RESULT_SCHEMA_VERSION,
    MLPerfSCValidationError,
    validate_mlperf_sc_result,
)


def _simulation_payload() -> dict[str, Any]:
    """A valid simulation-class MLPerf-SC payload with a relative artifact path."""
    return {
        "schema_version": MLPERF_SC_RESULT_SCHEMA_VERSION,
        "run": {
            "run_id": "shd-simulation-seed17",
            "task": "shd",
            "model": "dense-lif-frontier",
            "dataset": "shd",
            "started_at": "2026-05-18T03:20:00+02:00",
            "producer": "sc-neurocore",
        },
        "execution": {
            "backend": "python",
            "target": "cpu",
            "sc_mode": "bipolar",
            "bitstream_length": 1024,
            "seed": 17,
        },
        "metrics": {
            "accuracy": 0.869274,
            "latency_ms": 1.25,
            "throughput_inferences_per_s": 800.0,
            "energy_j_per_inference": None,
            "power_w": None,
            "area": {"luts": None, "ffs": None, "bram": None, "dsp": None},
        },
        "evidence": {
            "evidence_class": "simulation",
            "environment": {"python": "3.12.0", "platform": "linux-x86_64"},
            "artifacts": [
                {
                    "kind": "raw_results",
                    "path": "runs/shd-simulation-seed17.json",
                    "sha256": "a" * 64,
                }
            ],
        },
    }


_MUTATORS: list[tuple[Callable[[dict[str, Any]], Any], str]] = [
    (lambda p: p.__setitem__("schema_version", "x"), "schema_version must be"),
    (lambda p: p.pop("run"), "missing required keys"),
    (lambda p: p.__setitem__("surprise", 1), "has unsupported keys"),
    (lambda p: p.__setitem__("run", 123), "run must be a mapping"),
    (lambda p: p["run"].__setitem__("run_id", ""), "run.run_id must be a non-empty string"),
    (lambda p: p["execution"].__setitem__("sc_mode", "bogus"), "sc_mode has unsupported value"),
    (
        lambda p: p["execution"].__setitem__("bitstream_length", 0),
        "bitstream_length must be positive",
    ),
    (
        lambda p: p["execution"].__setitem__("bitstream_length", "x"),
        "bitstream_length must be an integer",
    ),
    (lambda p: p["execution"].__setitem__("seed", -1), "seed must be non-negative"),
    (lambda p: p["metrics"].__setitem__("accuracy", "x"), "accuracy must be numeric"),
    (lambda p: p["metrics"].__setitem__("latency_ms", math.inf), "latency_ms must be finite"),
    (lambda p: p["metrics"].__setitem__("latency_ms", -1.0), "latency_ms must be non-negative"),
    (
        lambda p: p["metrics"].__setitem__("throughput_inferences_per_s", 0.0),
        "must be positive when present",
    ),
    (lambda p: p["metrics"]["area"].__setitem__("luts", -1), "luts must be non-negative"),
    (lambda p: p["evidence"].__setitem__("environment", {}), "environment must not be empty"),
    (lambda p: p["evidence"].__setitem__("artifacts", "x"), "artifacts must be a sequence"),
    (lambda p: p["evidence"].__setitem__("artifacts", []), "artifacts must not be empty"),
    (
        lambda p: p["evidence"]["artifacts"][0].__setitem__("kind", "bogus"),
        "artifact.kind has unsupported value",
    ),
    (
        lambda p: p["evidence"]["artifacts"][0].__setitem__("sha256", "not-hex"),
        "sha256 must be 64 lowercase hex chars",
    ),
    (
        lambda p: p["metrics"].__setitem__("energy_j_per_inference", 0.5),
        "must not claim measured energy or power",
    ),
]


@pytest.mark.parametrize(("mutator", "match"), _MUTATORS)
def test_validate_rejects_malformed_payloads(
    mutator: Callable[[dict[str, Any]], Any], match: str
) -> None:
    """Each MLPerf-SC schema invariant rejects its specific malformed field."""
    payload = _simulation_payload()
    mutator(payload)

    with pytest.raises(MLPerfSCValidationError, match=match):
        validate_mlperf_sc_result(payload)


def test_board_measurement_requires_energy_or_power() -> None:
    """Board-measurement evidence with raw artifacts still needs an energy or power metric."""
    payload = _simulation_payload()
    payload["evidence"]["evidence_class"] = "board_measurement"
    payload["evidence"]["artifacts"][0]["kind"] = "board_log"

    with pytest.raises(MLPerfSCValidationError, match="requires energy or power"):
        validate_mlperf_sc_result(payload)


def test_synthesis_estimate_requires_report_artifacts() -> None:
    """Synthesis-estimate evidence requires a synthesis or ASIC report artifact."""
    payload = _simulation_payload()
    payload["evidence"]["evidence_class"] = "synthesis_estimate"

    with pytest.raises(MLPerfSCValidationError, match="requires synthesis or ASIC report"):
        validate_mlperf_sc_result(payload)


def test_validate_accepts_populated_area_metrics() -> None:
    """A payload with concrete area integers validates and returns a typed result."""
    payload = _simulation_payload()
    payload["metrics"]["area"] = {"luts": 128, "ffs": 256, "bram": 0, "dsp": 4}

    result = validate_mlperf_sc_result(payload)

    assert result.metrics.area.luts == 128
    assert result.metrics.area.bram == 0


def test_validate_reports_missing_artifact_file(tmp_path: Path) -> None:
    """With an artifact root, an artifact path that does not exist is rejected."""
    payload = _simulation_payload()

    with pytest.raises(MLPerfSCValidationError, match="does not exist"):
        validate_mlperf_sc_result(payload, artifact_root=tmp_path)


def test_validate_rejects_symlink_escape_from_artifact_root(tmp_path: Path) -> None:
    """An artifact path resolving outside the root via a symlink is rejected."""
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "evil").symlink_to(outside, target_is_directory=True)
    payload = _simulation_payload()
    payload["evidence"]["artifacts"][0]["path"] = "evil/leak.json"

    with pytest.raises(MLPerfSCValidationError, match="escapes artifact_root"):
        validate_mlperf_sc_result(payload, artifact_root=root)
