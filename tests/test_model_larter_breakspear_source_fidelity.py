# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Larter-Breakspear primary-equation and preservation evidence

"""Independent equation checks for the source and retained SC identities."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import tomllib

import pytest

from sc_neurocore.neurons.models import (
    LarterBreakspearNeuron,
    SCDecoupledAdaptationIonMassNeuron,
)

_ROOT = Path(__file__).resolve().parents[1]


def _source_derivatives(
    neuron: LarterBreakspearNeuron, v: float, w: float, z: float, coupling: float
) -> tuple[float, float, float]:
    def gate(value: float, threshold: float, width: float) -> float:
        return 0.5 * (1.0 + math.tanh((value - threshold) / width))

    m_ca = gate(v, neuron.t_ca, neuron.delta_ca)
    m_na = gate(v, neuron.t_na, neuron.delta_na)
    m_k = gate(v, neuron.t_k, neuron.delta_k)
    q_v = neuron.q_v_max * gate(v, neuron.v_t, neuron.delta_v)
    q_z = neuron.q_z_max * gate(z, neuron.z_t, neuron.delta_z)
    excitation = neuron.a_ee * (
        (1.0 - neuron.coupling_balance) * q_v + neuron.coupling_balance * coupling
    )
    dv = (
        -(neuron.g_ca + neuron.r_nmda * excitation) * m_ca * (v - neuron.v_ca)
        - neuron.g_k * w * (v - neuron.v_k)
        - neuron.g_l * (v - neuron.v_l)
        - (neuron.g_na * m_na + excitation) * (v - neuron.v_na)
        - neuron.a_ie * z * q_z
        + neuron.a_ne * neuron.i_ext
    )
    dw = neuron.phi * (m_k - w) / neuron.tau_k
    dz = neuron.b * (neuron.a_ni * neuron.i_ext + neuron.a_ei * v * q_v)
    return tuple(neuron.t_scale * value for value in (dv, dw, dz))


def test_source_derivatives_match_primary_equations() -> None:
    neuron = LarterBreakspearNeuron(v=-0.23, w=0.31, z=0.17)
    assert neuron._derivatives(neuron.v, neuron.w, neuron.z, 0.42) == pytest.approx(
        _source_derivatives(neuron, neuron.v, neuron.w, neuron.z, 0.42), abs=1.0e-15
    )


def test_inhibitory_state_feeds_back_into_source_voltage() -> None:
    low = LarterBreakspearNeuron(z=0.0)
    high = LarterBreakspearNeuron(z=0.5)
    assert (
        high._derivatives(high.v, high.w, high.z, 0.0)[0]
        < low._derivatives(low.v, low.w, low.z, 0.0)[0]
    )


def test_source_rk4_matches_independent_step() -> None:
    neuron = LarterBreakspearNeuron()
    state = (neuron.v, neuron.w, neuron.z)
    coupling = 0.37
    dt = neuron.dt
    k1 = _source_derivatives(neuron, *state, coupling)
    k2 = _source_derivatives(neuron, *(state[i] + 0.5 * dt * k1[i] for i in range(3)), coupling)
    k3 = _source_derivatives(neuron, *(state[i] + 0.5 * dt * k2[i] for i in range(3)), coupling)
    k4 = _source_derivatives(neuron, *(state[i] + dt * k3[i] for i in range(3)), coupling)
    expected = tuple(
        state[i] + dt * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0 for i in range(3)
    )
    neuron.step(coupling)
    assert (neuron.v, neuron.w, neuron.z) == pytest.approx(expected, abs=1.0e-15)


def test_old_project_recurrence_is_preserved_under_sc_identity() -> None:
    neuron = SCDecoupledAdaptationIonMassNeuron()
    assert neuron.step(0.0) == pytest.approx(-0.4987593419078305, abs=1.0e-15)
    assert (neuron.w, neuron.z) == pytest.approx(
        (0.0002412377194581311, 6.202934920494744e-07), abs=1.0e-15
    )


def test_source_and_sc_identities_are_distinct_and_public() -> None:
    assert LarterBreakspearNeuron is not SCDecoupledAdaptationIonMassNeuron
    assert LarterBreakspearNeuron().step(0.0) != SCDecoupledAdaptationIonMassNeuron().step(0.0)


@pytest.mark.parametrize("name", ["larter_breakspear", "sc_decoupled_adaptation_ion_mass"])
def test_toml_and_json_schemas_are_identical(name: str) -> None:
    root = _ROOT / "src/sc_neurocore/neurons/model_schemas"
    with (root / f"{name}.toml").open("rb") as stream:
        toml_payload = tomllib.load(stream)
    json_payload = json.loads((root / f"{name}.json").read_text(encoding="utf-8"))
    assert toml_payload == json_payload


@pytest.mark.parametrize(
    ("model", "receipt_name"),
    [
        (LarterBreakspearNeuron, "larter_breakspear_2003.json"),
        (SCDecoupledAdaptationIonMassNeuron, "sc_decoupled_adaptation_ion_mass.json"),
    ],
)
def test_mixed_drive_matches_frozen_receipt(model: type[object], receipt_name: str) -> None:
    receipt = json.loads(
        (_ROOT / "src/sc_neurocore/neurons/reference_receipts" / receipt_name).read_text(
            encoding="utf-8"
        )
    )
    neuron = model()
    digest = hashlib.sha256()
    pattern = receipt["drive"]["pattern"]
    for index in range(receipt["oracle"]["steps"]):
        neuron.step(pattern[index % len(pattern)])
        digest.update(struct.pack("<ddd", neuron.v, neuron.w, neuron.z))
    assert [neuron.v, neuron.w, neuron.z] == pytest.approx(
        receipt["oracle"]["final_state"], abs=2e-14
    )
    assert digest.hexdigest() == receipt["oracle"]["trace_sha256"]
