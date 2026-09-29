# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — the Studio clocks a run by what one step() call lasts

"""A macro-stepping model is timed by its macro step, not by its sub-step.

Hodgkin-Huxley, Connor-Stevens and Wang-Buzsaki advance a fixed macro step of
several ``dt`` sub-steps per ``step()`` call, as their profiles declare
(``numerical.macro_step``). The Studio counted each call as one ``dt``: a
100 ms Hodgkin-Huxley request ran 10 s of model time and reported 6330 Hz
instead of about 60 Hz, and Wang-Buzsaki at 10 µA/cm² reported 14 280 Hz.
"""

from __future__ import annotations

import pytest

from sc_neurocore.studio.experiment_spec import resolve_experiment, run_experiment
from sc_neurocore.studio.model_run_contract import (
    ModelInputError,
    declared_macro_step,
    macro_step_refusal,
    resolve_model_run_inputs,
)
from sc_neurocore.studio.network_graph import available_models, create_population
from sc_neurocore.studio.network_graph_spec import validate_graph

MACRO = {
    "HodgkinHuxleyNeuron": (0.01, 1.0),
    "ConnorStevensNeuron": (0.01, 1.0),
    "WangBuzsakiNeuron": (0.01, 0.5),
}


@pytest.mark.parametrize("name", sorted(MACRO))
def test_the_profile_declares_the_macro_step(name: str) -> None:
    assert declared_macro_step(name) == MACRO[name]


@pytest.mark.parametrize(
    "name",
    [
        "AdExNeuron",  # one dt per call
        "COBALIFNeuron",  # four stages of one scheme folded into one dt
        "SCWBNMDAMagnesiumBlockNeuron",  # descriptive profile, no runtime macro step
    ],
)
def test_a_model_stepping_once_per_dt_declares_no_macro_step(name: str) -> None:
    assert declared_macro_step(name) is None
    inputs = resolve_model_run_inputs(name, {}, None)
    assert inputs.step_ms == inputs.dt


@pytest.mark.parametrize("name", sorted(MACRO))
def test_the_run_clock_is_the_macro_step(name: str) -> None:
    sub_dt, macro = MACRO[name]
    result = run_experiment(resolve_experiment({"name": name, "duration": 20.0, "current": 10.0}))
    assert result["dt"] == macro
    assert result["n_steps"] == round(20.0 / macro)
    assert result["time"][-1] == pytest.approx(20.0)
    assert result["effective_inputs"]["dt"] == sub_dt
    assert result["effective_inputs"]["step_ms"] == macro
    assert result["experiment"]["numerical"]["step_ms"] == macro


def test_hodgkin_huxley_fires_at_a_physiological_rate() -> None:
    # The squid axon at 10 µA/cm² fires repetitively at roughly 60–70 Hz;
    # the Studio reported 6330 Hz while it counted sub-steps as time.
    result = run_experiment(
        resolve_experiment({"name": "HodgkinHuxleyNeuron", "duration": 500.0, "current": 10.0})
    )
    assert 40.0 <= result["stats"]["rate_hz"] <= 100.0


@pytest.mark.parametrize("name", sorted(MACRO))
def test_another_dt_is_refused_because_nothing_declares_its_step(name: str) -> None:
    sub_dt, macro = MACRO[name]
    with pytest.raises(ModelInputError) as info:
        resolve_model_run_inputs(name, {}, sub_dt * 5)
    assert info.value.field == "dt"
    assert f"{macro:g} ms macro step" in info.value.reason


def test_a_duration_shorter_than_one_macro_step_has_no_complete_step() -> None:
    with pytest.raises(Exception, match="yields no complete step"):
        resolve_experiment({"name": "HodgkinHuxleyNeuron", "duration": 0.5})


@pytest.mark.parametrize("name", sorted(MACRO))
def test_a_macro_stepping_model_cannot_join_a_network(name: str) -> None:
    assert macro_step_refusal(name) is not None
    assert name not in available_models()
    population = create_population(label="P", model=name, count=2)
    population["id"] = "p"
    graph = {"populations": [population], "projections": [], "duration": 5.0, "dt": MACRO[name][0]}
    errors = validate_graph(graph)
    assert len(errors) == 1, errors
    assert "macro step" in errors[0]
