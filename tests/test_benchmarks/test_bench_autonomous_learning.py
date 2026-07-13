# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Autonomous-learning benchmark tests

"""Test source binding, probe validation, and deterministic comparisons."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_BENCHMARKS = _ROOT / "benchmarks"
_RESULT = _BENCHMARKS / "results/bench_learning_bridge.json"
sys.path.insert(0, str(_BENCHMARKS))
support: Any = importlib.import_module("_learning_benchmark_support")
probe: Any = importlib.import_module("_learning_bridge_probe")


def _tree(root: Path) -> Path:
    bridge = root / "src/sc_neurocore/_native/learning_bridge.py"
    bridge.parent.mkdir(parents=True)
    bridge.write_text("VALUE = 1\n", encoding="utf-8")
    library = root / "libautonomous_learning.so"
    library.write_bytes(b"native")
    return library


def _args(tmp_path: Path) -> argparse.Namespace:
    baseline = tmp_path / "baseline"
    candidate = tmp_path / "candidate"
    return argparse.Namespace(
        iterations=2,
        steps=32,
        warmups=1,
        baseline_label="parent",
        candidate_label="candidate",
        baseline_ref="abc",
        candidate_ref="working-tree",
        baseline_root=baseline,
        candidate_root=candidate,
        baseline_lib=_tree(baseline),
        candidate_lib=_tree(candidate),
        cpu=None,
        no_affinity=True,
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("iterations", 0, "iterations and steps"),
        ("steps", 0, "iterations and steps"),
        ("warmups", -1, "warmups"),
        ("candidate_label", "parent", "labels"),
        ("baseline_ref", "", "references"),
    ],
)
def test_validate_args_rejects_invalid_configuration(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    """Configuration errors fail before any probe is executed."""
    args = _args(tmp_path)
    setattr(args, field, value)
    with pytest.raises(ValueError, match=message):
        support.validate_args(args)


def test_validate_args_requires_source_and_library(tmp_path: Path) -> None:
    """Both variants must bind real source and native-library artifacts."""
    args = _args(tmp_path)
    args.baseline_lib.unlink()
    with pytest.raises(ValueError, match="library does not exist"):
        support.validate_args(args)
    args.baseline_lib.write_bytes(b"native")
    (args.candidate_root / "src/sc_neurocore/_native/learning_bridge.py").unlink()
    with pytest.raises(ValueError, match="no learning bridge"):
        support.validate_args(args)


def test_source_digest_is_deterministic_and_content_bound(tmp_path: Path) -> None:
    """Source identity changes with implementation bytes, not invocation path."""
    root = tmp_path / "tree"
    _tree(root)
    first = support.source_digest(root)
    assert first == support.source_digest(root)
    bridge = root / "src/sc_neurocore/_native/learning_bridge.py"
    bridge.write_text("VALUE = 2\n", encoding="utf-8")
    second = support.source_digest(root)
    assert first[0] != second[0]
    assert set(second[1]) == {"src/sc_neurocore/_native/learning_bridge.py"}


def _payload() -> dict[str, Any]:
    timings = {metric: 10 for metric in support.TIMINGS[1:]}
    outputs = {field: 0.5 for field in support.WEIGHTS}
    outputs.update({field: "a" * 64 for field in support.HASHES})
    return {
        "timings": timings,
        "outputs": outputs,
        "canonical_sha256": "b" * 64,
        "canonical_bytes": 128,
        "max_rss_kib": 2048,
    }


def _completed(payload: object, *, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    stdout = payload if isinstance(payload, str) else json.dumps(payload)
    return subprocess.CompletedProcess(["probe"], returncode, stdout, "failure")


def test_sample_accepts_valid_nullable_language_timing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Optional Go and Julia absence remains explicit rather than fabricated."""
    root = tmp_path / "tree"
    library = _tree(root)
    payload = _payload()
    payload["timings"]["go_process_ns"] = None
    payload["outputs"]["go_weight"] = None
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs: _completed(payload))
    result = support.sample(support.Variant("test", root, library, "ref"), 8, None, None)
    assert result["go_process_ns"] is None
    assert result["go_weight"] is None
    assert result["subprocess_wall_ns"] > 0


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: payload["timings"].update(import_ns=0),
        lambda payload: payload["outputs"].update(rust_scalar_weight=float("nan")),
        lambda payload: payload["outputs"].update(layer_state_sha256="short"),
        lambda payload: payload.update(canonical_bytes=True),
    ],
)
def test_sample_rejects_malformed_payloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: Any
) -> None:
    """Malformed timing, output, hash, and count fields cannot enter evidence."""
    root = tmp_path / "tree"
    library = _tree(root)
    payload = _payload()
    mutation(payload)
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs: _completed(payload))
    with pytest.raises(RuntimeError, match="probe emitted invalid"):
        support.sample(support.Variant("test", root, library, "ref"), 8, None, None)


def test_sample_rejects_process_and_json_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Subprocess failures and non-JSON output retain actionable diagnostics."""
    root = tmp_path / "tree"
    variant = support.Variant("test", root, _tree(root), "ref")
    monkeypatch.setattr(
        support.subprocess, "run", lambda *args, **kwargs: _completed({}, returncode=4)
    )
    with pytest.raises(RuntimeError, match="probe failed: failure"):
        support.sample(variant, 8, None, None)
    monkeypatch.setattr(support.subprocess, "run", lambda *args, **kwargs: _completed("not-json"))
    with pytest.raises(RuntimeError, match="invalid JSON"):
        support.sample(variant, 8, None, None)


def _sample_value(timing: int = 10) -> dict[str, Any]:
    result = {metric: timing for metric in support.TIMINGS}
    result.update({field: 0.5 for field in support.WEIGHTS})
    result.update({field: "a" * 64 for field in support.HASHES})
    result.update(canonical_sha256="b" * 64, canonical_bytes=128, max_rss_kib=2048)
    return result


def test_summarize_and_compare_stable_samples() -> None:
    """Stable outputs produce timing summaries and parent/candidate deltas."""
    baseline = support.summarize([_sample_value(10), _sample_value(20)])
    candidate = support.summarize([_sample_value(20), _sample_value(40)])
    comparison = support.compare(baseline, candidate)
    assert baseline["rust_scalar_ns"]["median"] == 15
    assert comparison["canonical_output_equivalent"] is True
    assert comparison["rust_scalar_ns"]["candidate_delta_percent"] == pytest.approx(100.0)


def test_summarize_and_compare_reject_output_drift() -> None:
    """Within-variant and cross-variant deterministic drift both fail closed."""
    changed = _sample_value()
    changed["rust_scalar_weight"] = 0.6
    with pytest.raises(RuntimeError, match="changed rust_scalar_weight"):
        support.summarize([_sample_value(), changed])
    baseline = support.summarize([_sample_value()])
    candidate = support.summarize([_sample_value()])
    candidate["layer_state_sha256"] = "c" * 64
    with pytest.raises(RuntimeError, match="layer_state_sha256"):
        support.compare(baseline, candidate)


def test_probe_events_are_deterministic_and_typed() -> None:
    """Every language receives the same contiguous periodic event stream."""
    first = probe._events(64)
    second = probe._events(64)
    assert all(np.array_equal(left, right) for left, right in zip(first, second, strict=True))
    assert first[0].dtype == np.bool_
    assert first[1].flags.c_contiguous
    assert first[2].dtype == np.float32


def test_committed_evidence_matches_candidate_source() -> None:
    """Committed evidence remains bound to the current multi-language surface."""
    raw = _RESULT.read_text(encoding="utf-8")
    assert "/home/" not in raw and "/media/" not in raw and "/tmp/" not in raw
    payload: object = json.loads(raw)
    assert isinstance(payload, dict)
    assert payload["schema_version"] == ("sc-neurocore.learning-bridge-modularisation-benchmark.v1")
    configuration = payload["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["iterations"] == 5
    assert configuration["warmups"] == 1
    assert configuration["steps"] == 1024
    variants = payload["variants"]
    assert isinstance(variants, list)
    candidate = next(
        item
        for item in variants
        if isinstance(item, dict) and item.get("label") == "learning-modular"
    )
    digest, hashes = support.source_digest(_ROOT)
    assert candidate["source_sha256"] == digest == payload["source_sha256"]
    assert candidate["source_file_count"] == len(hashes) == 16
    assert payload["comparison"]["canonical_output_equivalent"] is True
    for language in ("go_process_ns", "julia_process_ns"):
        assert payload["comparison"][language]["available"] is True
