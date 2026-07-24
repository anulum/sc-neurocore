# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (stochastic_rejects) from former test_verilog_compiler_contracts.py

from __future__ import annotations

from tests.verilog_compiler_contracts_support import *  # noqa: F403

def test_stochastic_compilation_rejects_mutated_missing_probability() -> None:
    """Poisson compilation fails closed if its public expression is cleared."""
    neuron = EquationNeuron(
        equations={"v": "I"},
        state={"v": 0.0},
        detection="poisson",
        probability_expression="0.25",
    )
    neuron.probability_expression = None

    with pytest.raises(ValueError, match="requires a probability expression"):
        compile_to_verilog(neuron)


def test_stochastic_compilation_rejects_mutated_missing_rate() -> None:
    """Escape-rate compilation fails closed if its public expression is cleared."""
    neuron = _escape_rate_neuron()
    neuron.rate_expression = None

    with pytest.raises(ValueError, match="requires a rate expression"):
        compile_to_verilog(neuron)


def test_stochastic_compilation_rejects_mutated_missing_rng() -> None:
    """Registered stochastic RTL refuses a model whose RNG was removed."""
    neuron = _escape_rate_neuron()
    neuron._stochastic_rng = None

    with pytest.raises(ValueError, match="has no initial RNG seed"):
        compile_to_verilog(neuron)
