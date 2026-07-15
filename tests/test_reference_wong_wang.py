# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent Wong-Wang Appendix reference contract

"""Re-derive the enrolled Euler/OU trace without production equation helpers."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.wong_wang import WongWangUnit
from sc_neurocore.neurons.universal_dsl import load_schema

_ARTIFACT = (
    Path(__file__).resolve().parents[1]
    / "src/sc_neurocore/neurons/reference_trace_data/wong_wang_appendix_euler_ou_doi.json"
)
_STATE_NAMES = ("s1", "s2", "noise1", "noise2", "r1", "r2")


def _load() -> dict[str, Any]:
    """Load the committed reference artefact."""
    payload: object = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return cast("dict[str, Any]", payload)


def _independent_trace(steps: int) -> npt.NDArray[np.float64]:
    """Evaluate the cited reduced equations with direct scalar arithmetic."""
    s1 = 0.1
    s2 = 0.1
    noise1 = 0.0
    noise2 = 0.0
    dt = 0.0001
    tau_s = 0.1
    tau_ampa = 0.002
    gamma = 0.641
    j_n = 0.2609
    j_cross = 0.0497
    i_0 = 0.3255
    sigma = 0.02
    noise_scale = math.sqrt(dt / tau_ampa) * sigma
    rows = []

    def response(current: float) -> float:
        x = 270.0 * current - 108.0
        scaled = -0.154 * x
        if scaled > 700.0:
            return 0.0
        if abs(x) < 1.0e-7:
            return 1.0 / 0.154
        return max(0.0, x / -math.expm1(scaled))

    for step in range(steps):
        stim1 = 0.02 + 0.01 * math.sin(step * 0.07)
        stim2 = -0.01 + 0.008 * math.cos(step * 0.11)
        xi1 = math.sin((2 * step) * 0.17)
        xi2 = math.sin((2 * step + 1) * 0.17)
        rate1 = response(j_n * s1 - j_cross * s2 + i_0 + stim1 + noise1)
        rate2 = response(j_n * s2 - j_cross * s1 + i_0 + stim2 + noise2)
        next_s1 = s1 + dt * (-s1 / tau_s + (1.0 - s1) * gamma * rate1)
        next_s2 = s2 + dt * (-s2 / tau_s + (1.0 - s2) * gamma * rate2)
        next_noise1 = noise1 - (dt / tau_ampa) * noise1 + noise_scale * xi1
        next_noise2 = noise2 - (dt / tau_ampa) * noise2 + noise_scale * xi2
        s1, s2, noise1, noise2 = next_s1, next_s2, next_noise1, next_noise2
        rows.append((s1, s2, noise1, noise2, rate1, rate2))
    return np.asarray(rows, dtype="<f8")


def _features(trace: npt.NDArray[np.float64]) -> dict[str, float | str]:
    """Return the exact committed scalar and binary-digest feature set."""
    features: dict[str, float | str] = {}
    for index, name in enumerate(_STATE_NAMES):
        values = trace[:, index]
        features[f"first.{name}"] = float(values[0])
        features[f"final.{name}"] = float(values[-1])
        features[f"min.{name}"] = float(values.min())
        features[f"max.{name}"] = float(values.max())
        features[f"mean.{name}"] = float(values.mean())
    features["interleaved_f64le_sha256"] = hashlib.sha256(trace.tobytes(order="C")).hexdigest()
    return features


def test_reference_provenance_matches_paper_author_code_and_schema() -> None:
    """Pin the DOI, author-lab commit, equations, and paper timestep choice."""
    artefact = _load()
    provenance = artefact["provenance"]
    protocol = artefact["protocol"]
    schema = load_schema("wong_wang")
    assert provenance["citation"] == "doi:10.1523/JNEUROSCI.3733-05.2006"
    assert provenance["source_commit"] == "c39c6742329f89f1b5137f32910d55ad52d4bc24"
    assert "github.com/xjwanglab/wong-wang-2006" in provenance["source"]
    assert protocol["dt"] == schema["integration"]["dt"] == 0.0001
    assert schema["metadata"]["doi"] == "10.1523/JNEUROSCI.3733-05.2006"


def test_committed_features_match_independent_re_derivation() -> None:
    """Reproduce every feature and the binary trace digest independently."""
    artefact = _load()
    trace = _independent_trace(int(artefact["protocol"]["steps"]))
    expected = artefact["expected_features"]
    actual = _features(trace)
    assert set(actual) == set(expected)
    for key, value in actual.items():
        if isinstance(value, str):
            assert value == expected[key]
        else:
            assert value == pytest.approx(expected[key], abs=1.0e-15)


def test_python_model_matches_the_independent_trace() -> None:
    """Compare all physical states and pre-update rates, not summary statistics only."""
    steps = int(_load()["protocol"]["steps"])
    expected = _independent_trace(steps)
    unit = WongWangUnit()
    rows = []
    for step in range(steps):
        rates = unit.step_with_gaussian_samples(
            0.02 + 0.01 * math.sin(step * 0.07),
            -0.01 + 0.008 * math.cos(step * 0.11),
            math.sin((2 * step) * 0.17),
            math.sin((2 * step + 1) * 0.17),
        )
        rows.append((unit.s1, unit.s2, unit.noise1, unit.noise2, *rates))
    np.testing.assert_array_equal(np.asarray(rows, dtype="<f8"), expected)
