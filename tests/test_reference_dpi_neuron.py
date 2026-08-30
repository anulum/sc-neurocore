# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — DPI neuron independent reference contract

"""Independent reference-trace parity for DPI neuron."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.reference_traces import ReferenceTraceSpec, load_reference_trace_spec
from tests.cosim_support import _dpi_neuron_driven_euler_features


_PARITY_CASES: list[tuple[str, str, str, str, Callable[[ReferenceTraceSpec], dict[str, float]]]] = [
    (
        "dpi_neuron_driven_spiking_doi",
        "dpi_neuron",
        "independent_euler_reference",
        "doi:10.1109/ISCAS.2010.5536980",
        lambda spec: _dpi_neuron_driven_euler_features(
            current=spec.protocol.inputs["I"], dt=spec.protocol.dt, steps=spec.protocol.steps
        ),
    ),
]


def test_dpi_reference_rejects_empty_protocol() -> None:
    """Reject a reference request that cannot produce a final state."""
    with pytest.raises(ValueError, match="at least one step"):
        _dpi_neuron_driven_euler_features(current=5.0, dt=0.1, steps=0)


def _independent_source_orbit(
    receipt: dict[str, object],
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.uint8],
]:
    """Evaluate the DOI equations without importing production recurrence code."""
    initial = receipt["initial_state"]
    parameters = receipt["parameters"]
    protocol = receipt["protocol"]
    assert isinstance(initial, dict)
    assert isinstance(parameters, dict)
    assert isinstance(protocol, dict)
    state = {name: float(value) for name, value in initial.items()}
    values = {name: float(value) for name, value in parameters.items()}
    current = float(protocol["current"])
    steps = int(protocol["steps"])
    i_mem = np.empty(steps, dtype="<f8")
    i_ahp = np.empty(steps, dtype="<f8")
    refractory = np.empty(steps, dtype="<f8")
    events = np.empty(steps, dtype=np.uint8)

    for index in range(steps):
        pulse = state["refractory_time"] > 0.0
        spike_current = values["i_spike"] if pulse else 0.0
        d_i_ahp = (
            state["i_ahp"]
            / (values["tau_ahp"] * values["i_tau_ahp"])
            * (spike_current / (1.0 + state["i_ahp"] / values["i_ga"]) - values["i_tau_ahp"])
        )
        if pulse:
            d_i_mem = 0.0
        else:
            log_current = (math.log(values["i_0"]) + values["kappa"] * math.log(state["i_mem"])) / (
                values["kappa"] + 1.0
            )
            gate = 1.0 / (
                1.0 + math.exp(-values["alpha"] * (state["i_mem"] - values["i_threshold"]))
            )
            feedback = math.exp(log_current) * gate
            d_i_mem = (
                state["i_mem"]
                / (values["tau"] * values["i_tau"])
                * (
                    (values["i_rest"] + current) / (1.0 + state["i_mem"] / values["i_g"])
                    - values["i_tau"]
                    + feedback
                    - state["i_ahp"]
                )
            )

        next_i_ahp = state["i_ahp"] + values["dt"] * d_i_ahp
        if pulse:
            next_i_mem = values["i_reset"]
            next_refractory = max(0.0, state["refractory_time"] - values["dt"])
            event = 0
        else:
            next_i_mem = state["i_mem"] + values["dt"] * d_i_mem
            event = int(next_i_mem >= values["i_threshold"])
            next_refractory = 0.0
            if event:
                next_i_mem = values["i_reset"]
                next_refractory = values["refractory_period"]

        state = {
            "i_mem": next_i_mem,
            "i_ahp": next_i_ahp,
            "refractory_time": next_refractory,
        }
        i_mem[index] = next_i_mem
        i_ahp[index] = next_i_ahp
        refractory[index] = next_refractory
        events[index] = event

    return i_mem, i_ahp, refractory, events


def test_dpi_source_receipt_matches_independent_complete_orbit() -> None:
    """Bind the primary equations to every maintained state and event sample."""
    path = (
        Path(__file__).parents[1]
        / "src/sc_neurocore/neurons/reference_receipts/dpi_indiveri_stefanini_chicca_2010.json"
    )
    receipt = json.loads(path.read_text(encoding="utf-8"))
    protocol = receipt["protocol"]
    expected = receipt["expected"]
    parameters = receipt["parameters"]
    initial = receipt["initial_state"]
    steps = int(protocol["steps"])
    current = float(protocol["current"])
    oracle = _independent_source_orbit(receipt)
    inputs = np.full(steps, current, dtype="<f8")

    assert receipt["reference"]["pdf_sha256"] == (
        "7336974d3bf046e82f8b555b90e7f698bca7778cdf63229e9945a9cefb7a9807"
    )
    assert (
        hashlib.sha256(inputs.tobytes()).hexdigest()
        == (protocol["little_endian_float64_input_sha256"])
    )
    for trace, key in zip(
        oracle[:3],
        (
            "little_endian_float64_i_mem_sha256",
            "little_endian_float64_i_ahp_sha256",
            "little_endian_float64_refractory_sha256",
        ),
        strict=True,
    ):
        assert hashlib.sha256(trace.astype("<f8").tobytes()).hexdigest() == expected[key]
    assert hashlib.sha256(oracle[3].tobytes()).hexdigest() == expected["uint8_event_sha256"]
    assert np.flatnonzero(oracle[3]).tolist() == expected["event_indices_zero_based"]
    assert sum(int(value) for value in oracle[3]) == expected["event_count"]

    neuron = DPINeuron(**initial, **parameters)
    actual = neuron.simulate_complete(steps, current, "python")
    for actual_trace, oracle_trace in zip(actual, oracle, strict=True):
        np.testing.assert_array_equal(actual_trace, oracle_trace)
    assert {
        "i_mem": neuron.i_mem,
        "i_ahp": neuron.i_ahp,
        "refractory_time": neuron.refractory_time,
    } == expected["final_state"]


@pytest.mark.parametrize(
    ("trace_name", "schema_name", "kind", "citation", "reference"),
    _PARITY_CASES,
    ids=[case[1] for case in _PARITY_CASES],
)
def test_trace_features_match_independent_reference(
    trace_name: str,
    schema_name: str,
    kind: str,
    citation: str,
    reference: Callable[[ReferenceTraceSpec], dict[str, float]],
) -> None:
    """Each committed trace must reproduce an independent re-derivation to ``1e-12``.

    The per-case ``reference`` callable recomputes the expected feature map from the
    model's published equations (an explicit-Euler or analytic recurrence), so a
    passing assertion proves the committed corpus is independently reproduced rather
    than regenerated by the schema runner itself. The committed feature set must match
    the reference set exactly and every value to ``1e-12``.
    """
    spec = load_reference_trace_spec(trace_name)

    expected = reference(spec)

    assert spec.schema_name == schema_name
    assert spec.provenance.kind == kind
    assert spec.provenance.citation == citation
    assert set(expected) == set(spec.expected_features)
    for feature_name, feature_value in expected.items():
        assert spec.expected_features[feature_name] == pytest.approx(feature_value, abs=1e-12)
