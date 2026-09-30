# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — a sine drive may oscillate about a bias

"""A catalogue run's sine drive can carry a DC bias, and says so when it does."""

from __future__ import annotations

import pytest

from sc_neurocore.studio.model_run_contract import ModelInputError
from sc_neurocore.studio.model_simulate import simulate_model


def test_a_biased_sine_run_states_its_bias_and_stays_about_it() -> None:
    result = simulate_model(
        "AdExNeuron",
        dt=0.05,
        duration=50.0,
        current=500.0,
        protocol="sine",
        frequency_hz=40.0,
        bias=1000.0,
    )
    drive = result["current_trace"]
    assert min(drive) == pytest.approx(500.0, rel=0.01)
    assert max(drive) == pytest.approx(1500.0, rel=0.01)
    assert result["effective_inputs"]["bias"] == 1000.0


def test_a_run_without_a_bias_keeps_its_receipt_unchanged() -> None:
    receipt = simulate_model("AdExNeuron", dt=0.05, duration=5.0, current=1000.0)[
        "effective_inputs"
    ]
    assert "bias" not in receipt


def test_a_bias_on_another_protocol_is_refused_by_field() -> None:
    with pytest.raises(ModelInputError) as refused:
        simulate_model("AdExNeuron", dt=0.05, duration=5.0, current=1000.0, bias=10.0)
    assert refused.value.field == "bias"
    assert "sine protocol only" in str(refused.value)


def test_a_non_finite_bias_is_refused() -> None:
    with pytest.raises(ModelInputError) as refused:
        simulate_model("AdExNeuron", dt=0.05, duration=5.0, protocol="sine", bias=float("nan"))
    assert refused.value.field == "bias"
