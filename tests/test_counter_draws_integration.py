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

from sc_neurocore.neurons.equation_builder import from_equations
from sc_neurocore.neurons.model_profile import resolve_profile
from sc_neurocore.neurons.schema_validator import validate_schema_dict
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema
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


def test_playground_with_draws_and_xi_names_both_generators() -> None:
    """Equations with both kinds report the combined kind and both generators."""
    spec = resolve_ode_experiment(
        {"equations": ["dv/dt = -v + randn() + 0.1*xi"], "duration": 2.0, "seed": 3}
    )
    randomness = spec.public["randomness"]
    assert randomness["kind"] == "counter-draws+diffusion-noise"
    assert "Philox" in randomness["generator"] and "numpy" in randomness["generator"]


def test_playground_refuses_a_seed_for_equations_that_draw_nothing() -> None:
    """The refusal names both sources of randomness."""
    with pytest.raises(ExperimentRejected, match="no random draw"):
        resolve_ode_experiment({"equations": ["dv/dt = -v"], "duration": 2.0, "seed": 3})
