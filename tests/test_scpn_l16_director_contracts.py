# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — SCPN L16 Director cybernetic closure contract tests

"""Production contracts for the SCPN L16 Director stochastic layer."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.scpn.layers.l16_director import L16_DirectorLayer, L16_StochasticParameters


def test_l16_seed_scopes_output_bitstreams() -> None:
    params = L16_StochasticParameters(n_control_nodes=4, bitstream_length=128, rng_seed=1616)

    first = L16_DirectorLayer(params)
    second = L16_DirectorLayer(params)

    l15_input = {"gci": 0.8, "ethical_dissonance": 0.0, "free_energy": 0.0}
    first_step = first.step(0.01, l15_input=l15_input)
    second_step = second.step(0.01, l15_input=l15_input)
    next_step = first.step(0.01, l15_input=l15_input)

    np.testing.assert_array_equal(first_step["output_bitstreams"], second_step["output_bitstreams"])
    assert not np.array_equal(first_step["output_bitstreams"], next_step["output_bitstreams"])


def test_l16_meta_coupling_drives_recursive_hamiltonian_and_veto() -> None:
    layer = L16_DirectorLayer(
        L16_StochasticParameters(
            n_control_nodes=3,
            bitstream_length=32,
            kp=1.5,
            ki=0.5,
            veto_threshold=0.2,
            target_gci=0.8,
            integral_clamp=1.0,
            meta_coupling=0.5,
            rng_seed=5,
        )
    )

    out = layer.step(0.1, l15_input={"gci": 0.5, "ethical_dissonance": 0.4, "free_energy": 0.25})

    assert layer.integral_error == pytest.approx(0.05)
    assert out["control_signal"] == pytest.approx(0.775)
    assert out["entropy_flux"] == pytest.approx(0.625)
    assert out["entropy_proxy"] == pytest.approx(0.0625)
    assert out["h_rec"] == pytest.approx(0.7625)
    assert out["recursive_hamiltonian"] == pytest.approx(out["h_rec"])
    assert not out["veto_active"]
    assert np.all(out["qecc_syndrome"] == 0)
    assert np.all((out["will"] >= 0.0) & (out["will"] <= 1.0))


def test_l16_entropy_veto_zeroes_effective_will_and_syndrome() -> None:
    layer = L16_DirectorLayer(
        L16_StochasticParameters(
            n_control_nodes=2,
            bitstream_length=16,
            veto_threshold=0.05,
            meta_coupling=1.0,
            rng_seed=11,
        )
    )

    out = layer.step(0.1, l15_input={"gci": 0.0, "ethical_dissonance": 1.0, "free_energy": 1.0})

    assert out["veto_active"]
    assert np.all(out["effective_will"] == 0.0)
    assert np.all(out["qecc_syndrome"] == 1)
    assert not np.any(out["output_bitstreams"])


def test_l16_rejects_invalid_parameters_and_inputs() -> None:
    invalid_params = [
        {"n_control_nodes": 0},
        {"bitstream_length": 0},
        {"kp": float("nan")},
        {"ki": float("inf")},
        {"veto_threshold": -0.1},
        {"veto_threshold": 1.1},
        {"target_gci": -0.1},
        {"target_gci": 1.1},
        {"integral_clamp": 0.0},
        {"meta_coupling": -0.1},
        {"meta_coupling": 1.1},
        {"rng_seed": -1},
    ]
    for kwargs in invalid_params:
        with pytest.raises(ValueError):
            L16_DirectorLayer(L16_StochasticParameters(**kwargs))

    layer = L16_DirectorLayer(
        L16_StochasticParameters(n_control_nodes=2, bitstream_length=8, rng_seed=1)
    )

    invalid_steps = [
        (0.0, None),
        (float("nan"), None),
        (0.01, {"gci": -0.1}),
        (0.01, {"gci": 1.1}),
        (0.01, {"gci": float("nan")}),
        (0.01, {"gci": 0.5, "ethical_dissonance": 1.1}),
        (0.01, {"gci": 0.5, "free_energy": float("inf")}),
    ]
    for dt, l15_input in invalid_steps:
        with pytest.raises(ValueError):
            layer.step(dt, l15_input=l15_input)


def _l16_layer() -> L16_DirectorLayer:
    return L16_DirectorLayer(
        L16_StochasticParameters(n_control_nodes=4, bitstream_length=16, rng_seed=5)
    )


def test_l16_get_global_metric_and_default_step() -> None:
    layer = _l16_layer()
    assert isinstance(layer.get_global_metric(), float)
    # l15_input None takes the neutral-default step inputs.
    assert layer.step(0.01) is not None


def test_l16_step_requires_gci() -> None:
    layer = _l16_layer()
    with pytest.raises(ValueError, match="must include gci"):
        layer.step(0.01, l15_input={"ethical_dissonance": 0.1})


def test_l16_validate_boundary_context_branches() -> None:
    vbc = L16_DirectorLayer._validate_boundary_context
    # A null context id with empty terminals collapses to the neutral context.
    assert vbc({"boundary_context_id": None, "boundary_terminals": ()}) == (None, (), (), 1.0)
    # A null context id with non-iterable terminals is rejected.
    with pytest.raises(ValueError, match="non-empty sequence"):
        vbc({"boundary_context_id": None, "boundary_terminals": 5})
    # String terminals are rejected (a string is iterable but not a terminal list).
    with pytest.raises(ValueError, match="non-empty sequence"):
        vbc({"boundary_context_id": "ctx", "boundary_terminals": "T1T2"})
    # Non-iterable terminals with a real context id are rejected.
    with pytest.raises(ValueError, match="non-empty sequence"):
        vbc({"boundary_context_id": "ctx", "boundary_terminals": 5})
    # Duplicate terminals are rejected.
    with pytest.raises(ValueError, match="must not contain duplicates"):
        vbc({"boundary_context_id": "ctx", "boundary_terminals": ("T1", "T1")})
