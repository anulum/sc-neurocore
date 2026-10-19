# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — counter-based draws and graded spikes outside the compiler

"""The draws and graded spikes reach the schema layer, the profiles and the Studio playground.

A schema that calls ``rand()``, ``randn()`` or ``poisson()`` reports a
``philox-draws`` randomness contract with its seed and admits a seed override;
``threshold.spike_value`` makes a schema neuron graded and ``random_stream``
separates its instances; the schema validator knows the model's function
vocabulary; ``from_equations`` passes the seed, stream and graded value through;
and the Studio playground accepts a seed for equations that draw, replays it
and reports the Philox generator.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest
from starlette.testclient import TestClient

from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.schema_validator import validate_schema_dict
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
from sc_neurocore.studio.app import create_app
from sc_neurocore.studio.experiment_spec import (
    ExperimentRejected,
    resolve_ode_experiment,
    run_experiment,
)


def _noisy_lif(**threshold: Any) -> dict[str, Any]:
    """Return the bundled LIF schema with a randn() term and extra threshold keys."""
    schema = copy.deepcopy(load_schema("lif"))
    schema["dynamics"]["v"] = "-(v - v_rest) / tau_m + R * I / C + 2.0 * randn()"
    schema["threshold"].update(threshold)
    return schema


def test_profile_reports_philox_draws_with_their_seed() -> None:
    """A drawing schema is philox-draws, seeded by threshold.rng_seed and replayable."""
    randomness = resolve_profile(_noisy_lif(rng_seed=77), stem="lif").numerical.randomness
    assert randomness.kind == "philox-draws"
    assert randomness.seed == 77
    assert randomness.reproducible
    plain = resolve_profile(load_schema("lif"), stem="lif").numerical.randomness
    assert plain.kind == "none" and plain.seed is None


def test_schema_neuron_accepts_a_seed_override_and_separates_streams() -> None:
    """The seed override is admitted for a drawing schema; streams give different traces."""
    schema = _noisy_lif()
    traces = []
    for stream in (0, 0, 1):
        neuron = UniversalNeuron(schema, rng_seed_override=5, random_stream=stream)
        trace = []
        for _ in range(30):
            neuron.step(I=0.0)
            trace.append(neuron.to_equation_neuron().state["v"])
        traces.append(trace)
    assert traces[0] == traces[1]
    assert traces[0] != traces[2]


def test_schema_spike_value_makes_the_spike_graded() -> None:
    """threshold.spike_value is the value of each spike; 0 between spikes."""
    schema = copy.deepcopy(load_schema("lif"))
    schema["threshold"]["spike_value"] = "v - v_rest"
    neuron = UniversalNeuron(schema)
    values = []
    for _ in range(40):
        spike = neuron.step(I=20.0)
        values.append((spike, neuron.to_equation_neuron().last_spike_value))
    assert any(spike for spike, _ in values)
    assert all(value > 0.0 for spike, value in values if spike)
    assert all(value == 0.0 for spike, value in values if not spike)


def test_validator_knows_the_function_vocabulary() -> None:
    """rand/randn, sigmoid and a conditional expression raise no unknown-name warning."""
    schema = _noisy_lif()
    schema["dynamics"]["v"] = (
        "sigmoid(v) + rand() + (1.0 if v > 0.0 else 0.0) - (v - v_rest) / tau_m"
    )
    messages = [error.message for error in validate_schema_dict(schema, "lif")]
    assert not [message for message in messages if "unknown names" in message]
    schema["dynamics"]["v"] = "atan(v)"
    messages = [error.message for error in validate_schema_dict(schema, "lif")]
    assert any("unknown names: ['atan']" in message for message in messages)


def test_from_equations_passes_seed_stream_and_graded_value() -> None:
    """The Brian2-style builder forwards the draw key, the stream and spike_value."""
    neuron = from_equations(
        "dv/dt = -v + randn()",
        threshold="v > 1",
        reset="v = 0",
        rng_seed=9,
        random_stream=4,
        spike_value="v",
    )
    assert (neuron.random_seed, neuron.random_stream, neuron.spike_value_expr) == (9, 4, "v")


DRAWING = ["dv/dt = -v + 3.0*randn()"]


def test_playground_seed_keys_the_draws() -> None:
    """A seed is accepted for drawing equations, replays exactly and changes the trace."""
    spec = resolve_ode_experiment({"equations": DRAWING, "duration": 5.0, "seed": 11})
    randomness = spec.public["randomness"]
    assert randomness["kind"] == "counter-draws"
    assert randomness["seed"] == 11
    assert "Philox" in randomness["generator"]
    first = run_experiment(spec)["raw"]["states"]["v"]
    assert run_experiment(spec)["raw"]["states"]["v"] == first
    other = resolve_ode_experiment({"equations": DRAWING, "duration": 5.0, "seed": 12})
    assert run_experiment(other)["raw"]["states"]["v"] != first


def test_playground_with_draws_and_xi_names_one_generator() -> None:
    """Draws and xi report the combined kind; both come from the one Philox generator."""
    spec = resolve_ode_experiment(
        {"equations": ["dv/dt = -v + randn() + 0.1*xi"], "duration": 2.0, "seed": 3}
    )
    randomness = spec.public["randomness"]
    assert randomness["kind"] == "counter-draws+diffusion-noise"
    assert randomness["generator"] == "Philox4x32-10 counter draws keyed by the seed"


def test_playground_refuses_a_seed_for_equations_that_draw_nothing() -> None:
    """The refusal names both sources of randomness."""
    with pytest.raises(ExperimentRejected, match="no random draw"):
        resolve_ode_experiment({"equations": ["dv/dt = -v"], "duration": 2.0, "seed": 3})


# RIGOR review 275 (NEURO-41BD-RESET-DRAWS-01): a draw only in a reset assignment was not
# seen, so a seed was refused and an unseeded run was labelled deterministic and cacheable.
RESET_DRAW = {
    "equations": ["dv/dt = I"],
    "threshold": "v > 0.5",
    "reset": "v = rand()",
    "init": {"v": 0.0},
    "dt": 1.0,
    "duration": 8.0,
    "current": 1.0,
}


@pytest.mark.parametrize(
    ("equations", "reset"),
    [(["dv/dt = I"], "v = rand()"), (["dv/dt = I", "dw/dt = 0"], "w = 0.5; v = 0.25 * randn()")],
)
def test_a_draw_in_a_reset_assignment_takes_the_seed(equations: list[str], reset: str) -> None:
    """Reset-only and semicolon-separated mixed resets: seed accepted, replayed, and it matters."""
    request = {**RESET_DRAW, "equations": equations, "reset": reset}
    spec = resolve_ode_experiment({**request, "seed": 11})
    assert spec.public["randomness"]["kind"] == "counter-draws"
    assert spec.public["randomness"]["seed"] == 11
    first = run_experiment(spec)["raw"]["states"]["v"]
    assert run_experiment(spec)["raw"]["states"]["v"] == first
    other = resolve_ode_experiment({**request, "seed": 12})
    assert run_experiment(other)["raw"]["states"]["v"] != first


def test_a_fresh_trial_with_reset_draws_is_not_cached() -> None:
    """Without a seed the reset draws are disclosed, and a fresh trial is not cacheable."""
    replay = resolve_ode_experiment(RESET_DRAW)
    assert replay.public["randomness"]["kind"] == "counter-draws"
    assert replay.public["randomness"]["seed_source"] == "playground-default"
    fresh = resolve_ode_experiment({**RESET_DRAW, "trial": "fresh"})
    assert fresh.public["randomness"]["effective_trial"] == "fresh"
    assert fresh.cacheable is False


def test_the_http_route_runs_reset_draws() -> None:
    """POST /api/simulate: seed 11 is accepted and replays; two fresh trials differ."""
    client = TestClient(
        create_app(), base_url="http://127.0.0.1", headers={"sec-fetch-site": "none"}
    )
    seeded = [client.post("/api/simulate", json={**RESET_DRAW, "seed": 11}) for _ in range(2)]
    assert [response.status_code for response in seeded] == [200, 200]
    assert seeded[0].json()["experiment"]["randomness"]["seed"] == 11
    assert seeded[0].json()["raw"]["states"]["v"] == seeded[1].json()["raw"]["states"]["v"]
    fresh = [client.post("/api/simulate", json={**RESET_DRAW, "trial": "fresh"}) for _ in range(2)]
    assert [response.json()["cache"]["hit"] for response in fresh] == [False, False]
    assert fresh[0].json()["raw"]["states"]["v"] != fresh[1].json()["raw"]["states"]["v"]
