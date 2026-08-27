# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — independent NMDA equation and preservation evidence

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import tomllib

import pytest

from sc_neurocore.neurons.models import NMDANeuron, SCWBNMDAMagnesiumBlockNeuron

_ROOT = Path(__file__).resolve().parents[1]


def _source_derivatives(
    neuron: NMDANeuron, state: tuple[float, float, float, float], current: float
) -> tuple[float, float, float, float]:
    v, x_nmda, s_nmda, ca = state
    block = 1.0 / (1.0 + neuron.mg_conc * math.exp(-0.062 * v) / 3.57)
    return (
        (
            -neuron.g_l * (v - neuron.v_l)
            - neuron.g_ahp * ca * (v - neuron.v_k)
            - neuron.g_nmda * s_nmda * block * (v - neuron.e_nmda)
            + current
        )
        / neuron.c_m,
        neuron.kinetic_scale * (-x_nmda / neuron.tau_x),
        neuron.kinetic_scale * (neuron.alpha_s * x_nmda * (1.0 - s_nmda) - s_nmda / neuron.tau_s),
        -ca / neuron.tau_ca,
    )


def _source_step(
    neuron: NMDANeuron, current: float
) -> tuple[int, tuple[float, float, float, float, float]]:
    held = neuron.refractory_remaining > 0.0
    state = (
        neuron.v_reset if held else neuron.v,
        neuron.x_nmda,
        neuron.s_nmda,
        neuron.ca,
    )
    k1 = _source_derivatives(neuron, state, current)
    midpoint = tuple(state[index] + 0.5 * neuron.dt * k1[index] for index in range(4))
    k2 = _source_derivatives(neuron, midpoint, current)
    candidate = [state[index] + neuron.dt * k2[index] for index in range(4)]
    refractory = max(0.0, neuron.refractory_remaining - neuron.dt)
    event = 0
    if held:
        candidate[0] = neuron.v_reset
    elif candidate[0] >= neuron.v_threshold:
        event = 1
        candidate[0] = neuron.v_reset
        refractory = neuron.refractory_period
        candidate[1] += neuron.kinetic_scale * neuron.alpha_x
        candidate[3] += neuron.alpha_ca
    return event, (
        max(-120.0, min(80.0, candidate[0])),
        max(0.0, candidate[1]),
        max(0.0, min(1.0, candidate[2])),
        max(0.0, candidate[3]),
        refractory,
    )


def test_source_derivatives_match_wang_equations() -> None:
    neuron = NMDANeuron(v=-61.0, x_nmda=0.7, s_nmda=0.3, ca=0.2, g_ahp=0.05)
    state = (neuron.v, neuron.x_nmda, neuron.s_nmda, neuron.ca)
    assert neuron._derivatives(*state, 0.8) == pytest.approx(
        _source_derivatives(neuron, state, 0.8), abs=1.0e-15
    )


def test_source_step_matches_independent_midpoint_oracle() -> None:
    neuron = NMDANeuron(v=-53.0, x_nmda=0.4, s_nmda=0.2, ca=0.3, g_ahp=0.04)
    expected_event, expected_state = _source_step(neuron, 1.7)
    assert neuron.step(1.7) == expected_event
    assert (
        neuron.v,
        neuron.x_nmda,
        neuron.s_nmda,
        neuron.ca,
        neuron.refractory_remaining,
    ) == pytest.approx(expected_state, abs=1.0e-15)


def test_source_event_increment_enters_after_rk2_step() -> None:
    neuron = NMDANeuron(v=-52.01, x_nmda=0.25, s_nmda=0.1, ca=0.4)
    event, expected = _source_step(neuron, 1.0)
    assert event == 1
    neuron.step(1.0)
    assert neuron.x_nmda == pytest.approx(expected[1], abs=1.0e-15)
    assert neuron.ca == pytest.approx(expected[3], abs=1.0e-15)


def test_retained_project_recurrence_has_frozen_one_step_anchor() -> None:
    neuron = SCWBNMDAMagnesiumBlockNeuron()
    assert neuron.step(5.0) == 0
    assert (neuron.v, neuron.h, neuron.n, neuron.s_nmda) == pytest.approx(
        (-63.15566378039578, 0.6480311943997441, 0.237221887163776, 0.025),
        abs=1.0e-14,
    )


@pytest.mark.parametrize("name", ["nmda_neuron", "sc_wb_nmda_magnesium_block"])
def test_toml_and_json_schemas_are_identical(name: str) -> None:
    root = _ROOT / "src/sc_neurocore/neurons/model_schemas"
    with (root / f"{name}.toml").open("rb") as stream:
        toml_payload = tomllib.load(stream)
    json_payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    assert toml_payload == json_payload


@pytest.mark.parametrize(
    ("model", "receipt_name", "fields", "tolerance"),
    [
        (
            NMDANeuron,
            "nmda_neuron_wang_1999.json",
            ("v", "x_nmda", "s_nmda", "ca", "refractory_remaining"),
            2.0e-14,
        ),
        (
            SCWBNMDAMagnesiumBlockNeuron,
            "sc_wb_nmda_magnesium_block.json",
            ("v", "h", "n", "s_nmda"),
            2.0e-13,
        ),
    ],
)
def test_mixed_drive_matches_frozen_receipt(
    model: type[object], receipt_name: str, fields: tuple[str, ...], tolerance: float
) -> None:
    receipt = json.loads(
        (_ROOT / "src/sc_neurocore/neurons/reference_receipts" / receipt_name).read_text(
            encoding="utf-8"
        )
    )
    neuron = model()
    digest = hashlib.sha256()
    events = 0
    pattern = receipt["drive"]["pattern"]
    for index in range(receipt["oracle"]["steps"]):
        events += neuron.step(pattern[index % len(pattern)])
        values = tuple(getattr(neuron, field) for field in fields)
        digest.update(struct.pack("<" + "d" * len(values), *values))
    assert list(values) == pytest.approx(receipt["oracle"]["final_state"], abs=tolerance)
    assert events == receipt["oracle"]["event_count"]
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]


def test_source_and_component_citations_are_not_conflated() -> None:
    descriptor = tomllib.loads(
        (_ROOT / "src/sc_neurocore/neurons/model_descriptors/NMDANeuron.toml").read_text(
            encoding="utf-8"
        )
    )
    retained = tomllib.loads(
        (
            _ROOT / "src/sc_neurocore/neurons/model_descriptors/SCWBNMDAMagnesiumBlockNeuron.toml"
        ).read_text(encoding="utf-8")
    )
    assert descriptor["provenance"]["authors"] == ["Wang, X.-J."]
    assert descriptor["provenance"]["year"] == 1999
    assert retained["provenance"]["authors"] == ["SC-NeuroCore project"]
    assert "doi" not in retained["provenance"]
