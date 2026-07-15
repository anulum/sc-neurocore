# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Jansen–Rit equation-(6) reference contract

"""Re-derive the enrolled Euler trace without production equation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.jansen_rit import JansenRitUnit
from sc_neurocore.neurons.universal_dsl import load_schema

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/jansen_rit_eq6_euler_brian2.json"
)
_TRACE_NAMES = ("y0", "y3", "y1", "y4", "y2", "y5", "eeg")


def _load() -> dict[str, Any]:
    payload: object = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast("dict[str, Any]", payload)


def _independent_trace(steps: int) -> npt.NDArray[np.float64]:
    """Evaluate equation (6) with direct scalar simultaneous updates."""
    y0 = y3 = y1 = y4 = y2 = y5 = 0.0
    gain_a, gain_b = 3.25, 22.0
    rate_a, rate_b = 100.0, 50.0
    c1, c2, c3, c4 = 135.0, 108.0, 33.75, 33.75
    e0, v0, slope, dt = 2.5, 6.0, 0.56, 0.0001
    rows: list[tuple[float, ...]] = []

    def response(voltage: float) -> float:
        exponent = slope * (v0 - voltage)
        if exponent >= 0.0:
            exp_neg = math.exp(-exponent)
            return 2.0 * e0 * exp_neg / (1.0 + exp_neg)
        return 2.0 * e0 / (1.0 + math.exp(exponent))

    for index in range(steps):
        p_ext = 220.0 + 100.0 * math.sin(index * 0.037)
        s_pyramidal = response(y1 - y2)
        s_excitatory = response(c1 * y0)
        s_inhibitory = response(c3 * y0)
        candidate = (
            y0 + dt * y3,
            y3 + dt * (gain_a * rate_a * s_pyramidal - 2.0 * rate_a * y3 - rate_a**2 * y0),
            y1 + dt * y4,
            y4
            + dt
            * (gain_a * rate_a * (p_ext + c2 * s_excitatory) - 2.0 * rate_a * y4 - rate_a**2 * y1),
            y2 + dt * y5,
            y5 + dt * (gain_b * rate_b * c4 * s_inhibitory - 2.0 * rate_b * y5 - rate_b**2 * y2),
        )
        y0, y3, y1, y4, y2, y5 = candidate
        rows.append((*candidate, y1 - y2))
    return np.asarray(rows, dtype="<f8")


def _features(trace: npt.NDArray[np.float64]) -> dict[str, float | str]:
    features: dict[str, float | str] = {}
    for index, name in enumerate(_TRACE_NAMES):
        values = trace[:, index]
        features[f"first.{name}"] = float(values[0])
        features[f"final.{name}"] = float(values[-1])
        features[f"min.{name}"] = float(values.min())
        features[f"max.{name}"] = float(values.max())
        features[f"mean.{name}"] = float(values.mean())
    features["interleaved_f64le_sha256"] = hashlib.sha256(trace.tobytes(order="C")).hexdigest()
    return features


def test_reference_provenance_pins_equation_source_and_solver_scope() -> None:
    artefact = _load()
    provenance = artefact["provenance"]
    schema = load_schema("jansen_rit")
    assert provenance["citation"] == "doi:10.1007/BF00199471"
    assert provenance["source_commit"] == "1bfa1a9275bd9672b49f4bf61ffbaf6f7cb55fc9"
    assert "brian-team/brian2" in provenance["source"]
    assert "not a reproduction" in provenance["input_note"]
    assert artefact["protocol"]["dt"] == schema["integration"]["dt"] == 0.0001
    assert schema["metadata"]["doi"] == "10.1007/BF00199471"


def test_committed_features_match_independent_re_derivation() -> None:
    artefact = _load()
    trace = _independent_trace(int(artefact["protocol"]["steps"]))
    actual = _features(trace)
    expected = artefact["expected_features"]
    assert set(actual) == set(expected)
    for key, value in actual.items():
        if isinstance(value, str):
            assert value == expected[key]
        else:
            assert value == pytest.approx(expected[key], abs=1.0e-15)


def test_python_model_matches_every_independent_state() -> None:
    steps = int(_load()["protocol"]["steps"])
    expected = _independent_trace(steps)
    unit = JansenRitUnit()
    rows = []
    for index in range(steps):
        eeg = unit.step(220.0 + 100.0 * math.sin(index * 0.037))
        rows.append((unit.y0, unit.y3, unit.y1, unit.y4, unit.y2, unit.y5, eeg))
    np.testing.assert_array_equal(np.asarray(rows, dtype="<f8"), expected)
