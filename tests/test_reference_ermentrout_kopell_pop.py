# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent MPR equation-(12) reference contract

"""Re-derive the enrolled MPR Euler trace without production helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.ermentrout_kopell_pop import (
    ErmentroutKopellPopulation,
)
from sc_neurocore.neurons.universal_dsl import load_schema

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/ermentrout_kopell_pop_eq12_euler_doi.json"
)
_TRACE_NAMES = ("r", "v")


def _load() -> dict[str, Any]:
    payload: object = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast("dict[str, Any]", payload)


def _independent_trace(steps: int) -> npt.NDArray[np.float64]:
    """Restore physical variables from dimensionless equation (12)."""
    r, v = 0.1, -2.0
    tau, delta, eta_bar, coupling, dt = 1.0, 1.0, -5.0, 15.0, 0.01
    rows: list[tuple[float, float]] = []
    for index in range(steps):
        drive = 1.5 + 0.5 * math.sin(index * 0.037)
        next_r = r + dt * (delta / (math.pi * tau**2) + 2.0 * r * v / tau)
        next_v = v + dt * (
            (v**2 + eta_bar + drive + coupling * tau * r - (math.pi * tau * r) ** 2) / tau
        )
        r, v = next_r, next_v
        rows.append((r, v))
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
    artifact = _load()
    provenance = artifact["provenance"]
    schema = load_schema("ermentrout_kopell_pop")
    assert provenance["citation"] == "doi:10.1103/PhysRevX.5.021028"
    assert provenance["source"] == "https://doi.org/10.1103/PhysRevX.5.021028"
    assert "dimensionless equations" in provenance["equation"]
    assert "R=tau*r and t'=t/tau" in provenance["change_of_variables"]
    assert "not a paper requirement" in provenance["timestep_note"]
    assert "not a reproduction" in provenance["input_note"]
    assert "legacy public class name" in provenance["identity_note"]
    assert artifact["protocol"]["dt"] == schema["integration"]["dt"] == 0.01
    assert schema["metadata"]["doi"] == "10.1103/PhysRevX.5.021028"
    assert schema["extensions"]["source_change_of_variables"] == provenance["change_of_variables"]


def test_committed_features_match_independent_re_derivation() -> None:
    artifact = _load()
    trace = _independent_trace(int(artifact["protocol"]["steps"]))
    actual = _features(trace)
    expected = artifact["expected_features"]
    assert set(actual) == set(expected)
    for key, value in actual.items():
        if isinstance(value, str):
            assert value == expected[key]
        else:
            assert value == pytest.approx(expected[key], abs=1.0e-15)


def test_python_model_matches_every_independent_state() -> None:
    steps = int(_load()["protocol"]["steps"])
    expected = _independent_trace(steps)
    unit = ErmentroutKopellPopulation()
    rows = []
    for index in range(steps):
        drive = 1.5 + 0.5 * math.sin(index * 0.037)
        unit.step(drive)
        rows.append((unit.r, unit.v))
    np.testing.assert_array_equal(np.asarray(rows, dtype="<f8"), expected)
