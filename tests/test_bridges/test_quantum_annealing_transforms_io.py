# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Quantum-annealing transform and export tests

"""Exercise schedules, gauges, SC encodings, atomic JSON, and visualization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.bridges import annealing_backends as backends
from sc_neurocore.bridges import annealing_io
from sc_neurocore.bridges.quantum_annealing import (
    AnnealingSchedule,
    GaugeTransform,
    QUBOModel,
    SCPrecisionEncoder,
    export_bqm,
    export_ising_json,
    export_qubo_json,
    visualize_ising,
)
from tests.test_bridges.quantum_annealing_test_helpers import simple_ising, unsafe


def test_schedule_builders_and_defensive_points() -> None:
    """Every schedule is monotonic and exported through a defensive copy."""
    schedule = AnnealingSchedule()
    assert schedule.points == []
    assert schedule.total_time_us == 0.0

    assert schedule.linear(100.0) is schedule
    assert schedule.points == [(0.0, 0.0), (100.0, 1.0)]
    copied = schedule.points
    copied.append((200.0, 1.0))
    assert schedule.total_time_us == 100.0

    schedule.pause_and_quench(5.0, 0.4, 50.0, 1.0)
    assert schedule.points == [(0.0, 0.0), (5.0, 0.4), (55.0, 0.4), (56.0, 1.0)]
    schedule.reverse(1.0, 0.3, 5.0, 10.0, 5.0)
    assert schedule.to_dict() == {
        "schedule": [(0.0, 1.0), (5.0, 0.3), (15.0, 0.3), (20.0, 1.0)],
        "total_time_us": 20.0,
        "n_points": 4,
    }


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: AnnealingSchedule().linear(0.0), "duration_us"),
        (lambda: AnnealingSchedule().linear(float("nan")), "finite"),
        (lambda: AnnealingSchedule().pause_and_quench(ramp_time_us=0.0), "ramp"),
        (lambda: AnnealingSchedule().pause_and_quench(pause_duration_us=-1.0), "pause_duration"),
        (lambda: AnnealingSchedule().pause_and_quench(quench_time_us=0.0), "quench"),
        (lambda: AnnealingSchedule().pause_and_quench(pause_at_s=-0.1), "between"),
        (lambda: AnnealingSchedule().pause_and_quench(pause_at_s=0.0), "strictly"),
        (lambda: AnnealingSchedule().pause_and_quench(pause_at_s=1.0), "strictly"),
        (lambda: AnnealingSchedule().reverse(initial_s=2.0), "between"),
        (lambda: AnnealingSchedule().reverse(reverse_to_s=1.0), "smaller"),
        (lambda: AnnealingSchedule().reverse(ramp_time_us=0.0), "ramp"),
        (lambda: AnnealingSchedule().reverse(hold_time_us=0.0), "hold"),
        (lambda: AnnealingSchedule().reverse(forward_time_us=0.0), "forward"),
    ],
)
def test_schedule_rejects_invalid_points(call: object, match: str) -> None:
    """Non-finite, non-positive, and out-of-range schedule inputs fail."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def test_gauge_transform_preserves_model_structure() -> None:
    """Gauge copies retain topology, magnitudes, offset, and labels."""
    model = simple_ising()
    transformed = GaugeTransform(3, seed=42).transform(model)
    assert len(transformed) == 3
    for index, candidate in enumerate(transformed):
        assert candidate.n_qubits == model.n_qubits
        assert candidate.offset == model.offset
        assert candidate.qubit_labels == model.qubit_labels
        assert {pair: abs(value) for pair, value in candidate.J.items()} == {
            pair: abs(value) for pair, value in model.J.items()
        }
        assert candidate.source == f"test_gauge{index}"


def test_gauge_untransform_and_validation() -> None:
    """Spin reversal is its own inverse and validates both mappings."""
    transform = GaugeTransform(1)
    sample = {0: -1, 1: 1}
    gauge = {0: -1, 1: 1}
    assert transform.untransform_sample(sample, gauge) == {0: 1, 1: 1}
    assert transform.untransform_sample({0: 1}, {}) == {0: 1}
    with pytest.raises(ValueError, match="sample indices"):
        transform.untransform_sample({unsafe(-1): 1}, {})
    with pytest.raises(ValueError, match="sample values"):
        transform.untransform_sample({0: 0}, {})
    with pytest.raises(ValueError, match="gauge values"):
        transform.untransform_sample({0: 1}, {0: 2})


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: GaugeTransform(0), "positive"),
        (lambda: GaugeTransform(unsafe(True)), "positive"),
        (lambda: GaugeTransform(seed=unsafe(1.5)), "seed"),
        (lambda: GaugeTransform().transform(unsafe("bad")), "non-empty"),
    ],
)
def test_gauge_rejects_invalid_configuration(call: object, match: str) -> None:
    """Gauge counts, seeds, and models are validated."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


@pytest.mark.parametrize(
    ("encoding", "n_bits", "values", "tolerance"),
    [
        ("binary", 8, [0.0, 0.25, 0.5, 0.75, 1.0], 1 / 255),
        ("unary", 8, [0.0, 0.25, 0.5, 0.75, 1.0], 1 / 8),
        ("one_hot", 8, [0.0, 0.25, 0.5, 0.75, 1.0], 1 / 7),
    ],
)
def test_precision_encoding_roundtrips(
    encoding: str,
    n_bits: int,
    values: list[float],
    tolerance: float,
) -> None:
    """All supported encodings round-trip within their quantization step."""
    encoder = SCPrecisionEncoder(encoding, n_bits)
    for value in values:
        assert encoder.decode(encoder.encode(value)) == pytest.approx(value, abs=tolerance)
    assert encoder.qubits_needed(3) == 3 * n_bits
    assert len(encoder.encode_array(np.asarray(values))) == len(values) * n_bits


def test_precision_levels_clipping_and_empty_one_hot() -> None:
    """Level counts are exact, clipping is bounded, and empty one-hot decodes to zero."""
    assert SCPrecisionEncoder("binary", 4).n_levels == 16
    assert SCPrecisionEncoder("unary", 4).n_levels == 5
    assert SCPrecisionEncoder("one_hot", 4).n_levels == 4
    assert (
        SCPrecisionEncoder("binary", 4).decode(SCPrecisionEncoder("binary", 4).encode(-2.0)) == 0.0
    )
    assert (
        SCPrecisionEncoder("binary", 4).decode(SCPrecisionEncoder("binary", 4).encode(2.0)) == 1.0
    )
    assert SCPrecisionEncoder("one_hot", 1).decode({}) == 0.0


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: SCPrecisionEncoder("bad"), "Unknown encoding"),
        (lambda: SCPrecisionEncoder(n_bits=0), "positive"),
        (lambda: SCPrecisionEncoder().encode(float("nan")), "finite"),
        (lambda: SCPrecisionEncoder().decode({unsafe(-1): 1}), "indices"),
        (lambda: SCPrecisionEncoder().decode({0: 2}), "binary"),
        (lambda: SCPrecisionEncoder("one_hot", 3).decode({0: 1, 1: 1}), "at most one"),
        (lambda: SCPrecisionEncoder().qubits_needed(unsafe(True)), "non-negative"),
        (lambda: SCPrecisionEncoder().encode_array(np.ones((2, 2))), "one-dimensional"),
        (lambda: SCPrecisionEncoder().encode_array(np.array([])), "non-empty"),
        (lambda: SCPrecisionEncoder().encode_array(np.array([np.inf])), "finite"),
    ],
)
def test_precision_encoder_rejects_invalid_inputs(call: object, match: str) -> None:
    """Precision configuration, qubits, and arrays fail closed."""
    with pytest.raises(ValueError, match=match):
        unsafe(call)()


def test_json_exports_are_canonical_and_replace_atomically(tmp_path: Path) -> None:
    """Ising and QUBO JSON exports are deterministic UTF-8 files."""
    ising_path = tmp_path / "model.json"
    export_ising_json(simple_ising(), ising_path)
    first_payload = ising_path.read_bytes()
    assert first_payload.endswith(b"\n")
    data = json.loads(first_payload)
    assert data["type"] == "ising"
    assert data["J"] == {"0,1": -1.0, "1,2": 0.5}

    replacement = simple_ising()
    replacement.source = "replacement"
    export_ising_json(replacement, ising_path)
    assert json.loads(ising_path.read_text(encoding="utf-8"))["source"] == "replacement"

    qubo_path = tmp_path / "qubo.json"
    export_qubo_json(QUBOModel(Q={(1, 0): 2.0}, source="q"), qubo_path)
    assert json.loads(qubo_path.read_text(encoding="utf-8"))["Q"] == {"0,1": 2.0}
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_atomic_export_cleans_temporary_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An install failure leaves neither destination nor temporary residue."""

    def fail_replace(source: object, destination: object) -> None:
        raise OSError(f"cannot replace {source} with {destination}")

    monkeypatch.setattr(annealing_io, "_install_temporary", fail_replace)
    destination = tmp_path / "model.json"
    with pytest.raises(OSError, match="cannot replace"):
        export_ising_json(simple_ising(), destination)
    assert not destination.exists()
    assert not tuple(tmp_path.iterdir())


def test_bqm_export_optional_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """BQM export returns None without dimod and forwards spin data when present."""
    monkeypatch.setattr(backends, "HAS_DIMOD", False)
    assert export_bqm(simple_ising()) is None

    captured: tuple[object, ...] = ()

    class FakeDimod:
        @staticmethod
        def BinaryQuadraticModel(*args: object) -> str:
            nonlocal captured
            captured = args
            return "bqm"

    monkeypatch.setattr(backends, "HAS_DIMOD", True)
    monkeypatch.setattr(backends, "dimod", FakeDimod)
    model = simple_ising()
    assert export_bqm(model) == "bqm"
    assert captured == (model.h, model.J, model.offset, "SPIN")


def test_exports_reject_wrong_model_types(tmp_path: Path) -> None:
    """Serialization functions require the matching model type."""
    with pytest.raises(ValueError, match="IsingModel"):
        export_ising_json(unsafe("bad"), tmp_path / "x")
    with pytest.raises(ValueError, match="QUBOModel"):
        export_qubo_json(unsafe("bad"), tmp_path / "x")
    with pytest.raises(ValueError, match="IsingModel"):
        export_bqm(unsafe("bad"))
    with pytest.raises(ValueError, match="IsingModel"):
        visualize_ising(unsafe("bad"))


def test_visualization_contains_fields_and_couplings() -> None:
    """Text rendering names the model, biases, and coupling signs."""
    rendered = visualize_ising(simple_ising())
    assert "Ising Model: test" in rendered
    assert "Biases (h)" in rendered
    assert "Couplings (J)" in rendered
    assert "ferro" in rendered
    assert "anti" in rendered
