# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Poisson schema and stochastic-DSL contracts

"""Bind the stateless source schema to the hand model and emitted RTL."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from sc_neurocore.compiler.verilog_compiler import compile_to_datapath
from sc_neurocore.neurons.models.poisson import PoissonNeuron
from sc_neurocore.neurons.schema_module_aliases import resolve_schema_join
from sc_neurocore.neurons.universal_dsl import UniversalNeuron, load_schema

_REPOSITORY = Path(__file__).resolve().parents[1]


def _events(
    neuron: UniversalNeuron,
    steps: int,
    rate_override: float = -1.0,
) -> NDArray[np.uint8]:
    return np.fromiter(
        (neuron.step(I=rate_override) for _ in range(steps)),
        dtype=np.uint8,
        count=steps,
    )


def test_paired_schemas_encode_the_same_stateless_probability_contract() -> None:
    """TOML and JSON remain equivalent executable interchange forms."""
    source = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
    toml = load_schema(source / "poisson.toml")
    json_schema = load_schema(source / "poisson.json")
    assert toml == json_schema
    assert toml["state"] == {}
    assert toml["dynamics"] == {}
    assert toml["threshold"]["condition"] == "stochastic"
    assert toml["threshold"]["detection"] == "poisson"
    assert resolve_schema_join("poisson") == ("poisson", "PoissonNeuron")


def test_hand_toml_and_json_streams_are_exact_for_stored_and_override_rates() -> None:
    """Both input conventions consume the same seeded Bernoulli recurrence."""
    source = _REPOSITORY / "src/sc_neurocore/neurons/model_schemas"
    for rate_override in (-1.0, 400.0):
        hand = PoissonNeuron(rate_hz=250.0, dt_ms=1.0, seed=0xBEEF)
        expected = np.fromiter(
            (hand.step(rate_override) for _ in range(4096)),
            dtype=np.uint8,
            count=4096,
        )
        for schema_path in (source / "poisson.toml", source / "poisson.json"):
            schema = UniversalNeuron.from_schema(
                schema_path,
                parameter_overrides={"rate_hz": 250.0, "dt_ms": 1.0},
                rng_seed_override=0xBEEF,
            )
            actual = _events(schema, 4096, rate_override)
            np.testing.assert_array_equal(actual, expected)
            assert schema.state == {}
            assert schema.to_equation_neuron().stochastic_rng_state == hand.rng_state


def test_schema_reset_replays_rng_without_inventing_membrane_state() -> None:
    """Reset restores only the private stochastic execution state."""
    schema = UniversalNeuron.from_schema("poisson", rng_seed_override=42)
    first = _events(schema, 1000)
    assert schema.state == {}
    schema.reset()
    equation = schema.to_equation_neuron()
    assert equation.stochastic_rng_state == equation.stochastic_rng_initial_seed == 42
    np.testing.assert_array_equal(_events(schema, 1000), first)


def test_registered_and_folded_rtl_are_stateless_poisson_datapaths() -> None:
    """Both production emitters carry probability and RNG logic without phase state."""
    schema = UniversalNeuron.from_schema("poisson", rng_seed_override=0xACE1)
    registered = schema.to_verilog(
        module_name="sc_poisson_schema_contract",
        data_width=48,
        fraction=24,
    )
    folded = compile_to_datapath(
        schema.to_equation_neuron(),
        module_name="sc_poisson_folded_contract",
        data_width=48,
        fraction=24,
    )
    for rtl in (registered, folded):
        assert "spike_out" in rtl
        assert "_escape_probability" in rtl
        assert "_escape_threshold" in rtl
        assert "phase" not in rtl
    assert "_escape_lfsr" in registered
    assert "rng_sample" in folded
