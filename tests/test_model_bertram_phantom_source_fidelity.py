# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent Bertram phantom source audit

from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np
import pytest

from sc_neurocore.neurons.models.bertram_phantom import BertramPhantomBurster
from sc_neurocore.neurons.models.sc_three_state_phantom import SCThreeStatePhantomBurster

_ROOT = Path(__file__).resolve().parents[1]


def _sigmoid(v: float, midpoint: float, slope: float) -> float:
    return 1.0 / (1.0 + math.exp((midpoint - v) / slope))


def _independent_rhs(model: BertramPhantomBurster, state: np.ndarray, drive: float) -> np.ndarray:
    v, n, s1, s2 = state
    m_inf = _sigmoid(v, -22.0, 7.5)
    n_inf = _sigmoid(v, -9.0, 10.0)
    s1_inf = _sigmoid(v, -40.0, 0.5)
    s2_inf = _sigmoid(v, -42.0, 0.4)
    tau_n = 9.09 / (1.0 + math.exp((v + 9.0) / 10.0))
    currents = (
        280.0 * m_inf * (v - 100.0)
        + 1300.0 * n * (v + 80.0)
        + 20.0 * s1 * (v + 80.0)
        + 32.0 * s2 * (v + 80.0)
        + 25.0 * (v + 40.0)
    )
    return np.array(
        [
            (-currents + drive) / 4524.0,
            1.1 * (n_inf - n) / tau_n,
            (s1_inf - s1) / 1000.0,
            (s2_inf - s2) / 120_000.0,
        ]
    )


def test_author_code_defaults_and_four_state_identity() -> None:
    model = BertramPhantomBurster()
    assert (model.v, model.n, model.s1, model.s2) == (-43.0, 0.03, 0.1, 0.434)
    assert (model.g_ca, model.g_k, model.g_s1, model.g_s2, model.g_l) == (
        280.0,
        1300.0,
        20.0,
        32.0,
        25.0,
    )
    assert (model.c_m, model.tau_n_bar, model.tau_s1, model.tau_s2) == (
        4524.0,
        9.09,
        1000.0,
        120_000.0,
    )


def test_one_step_matches_independent_four_state_rk4() -> None:
    model = BertramPhantomBurster()
    state = np.array([model.v, model.n, model.s1, model.s2])
    drive = 17.0
    k1 = _independent_rhs(model, state, drive)
    k2 = _independent_rhs(model, state + 0.5 * model.dt * k1, drive)
    k3 = _independent_rhs(model, state + 0.5 * model.dt * k2, drive)
    k4 = _independent_rhs(model, state + model.dt * k3, drive)
    expected = state + model.dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
    model.step(drive)
    np.testing.assert_allclose([model.v, model.n, model.s1, model.s2], expected, rtol=0, atol=2e-14)


def test_dynamic_n_distinguishes_source_from_project_recurrence() -> None:
    source = BertramPhantomBurster()
    project = SCThreeStatePhantomBurster()
    initial_n = source.n
    source.step(0.0)
    project.step(0.0)
    assert source.n != initial_n
    assert not hasattr(project, "n")


@pytest.mark.parametrize("field", ["n", "s1", "s2"])
def test_corrupt_gate_is_rejected_before_mutation(field: str) -> None:
    model = BertramPhantomBurster()
    setattr(model, field, 1.1)
    before = (model.v, model.n, model.s1, model.s2)
    with pytest.raises(ValueError, match=field):
        model.step(0.0)
    assert (model.v, model.n, model.s1, model.s2) == before


def test_nonfinite_drive_is_atomic() -> None:
    model = BertramPhantomBurster()
    before = (model.v, model.n, model.s1, model.s2)
    with pytest.raises(ValueError, match="current"):
        model.step(float("nan"))
    assert (model.v, model.n, model.s1, model.s2) == before


def test_paired_schemas_preserve_source_identity() -> None:
    base = _ROOT / "src/sc_neurocore/neurons/model_schemas/bertram_phantom"
    with base.with_suffix(".toml").open("rb") as handle:
        toml_payload = tomllib.load(handle)
    json_payload = json.loads(base.with_suffix(".json").read_text(encoding="utf-8"))
    assert toml_payload == json_payload
    assert toml_payload["state"] == {"v": -43.0, "n": 0.03, "s1": 0.1, "s2": 0.434}
    assert toml_payload["metadata"]["doi"] == "10.1016/S0006-3495(00)76525-8"


def test_independent_receipt_reproduces_enrolled_trace() -> None:
    receipt_path = _ROOT / "src/sc_neurocore/neurons/reference_receipts/bertram_phantom_2000.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    model = BertramPhantomBurster()
    states: list[tuple[float, float, float, float]] = []
    events: list[int] = []
    for _ in range(receipt["drive"]["repeats"]):
        for current in receipt["drive"]["pattern"]:
            events.append(model.step(current))
            states.append((model.v, model.n, model.s1, model.s2))
    state_bytes = np.asarray(states, dtype="<f8").tobytes()
    event_bytes = np.asarray(events, dtype="<i8").tobytes()
    assert sum(events) == receipt["oracle"]["events"]
    assert (
        hashlib.sha256(state_bytes + event_bytes).hexdigest() == receipt["oracle"]["trace_sha256"]
    )
