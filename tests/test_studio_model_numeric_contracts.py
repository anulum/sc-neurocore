# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio offers and compiles only formats a model fits

"""Studio offers a Q-format only where the model's RTL holds the model.

The catalogue models, their schemas and the compile resolution are the
production ones.
"""

from __future__ import annotations

import pytest

from sc_neurocore.neurons.universal_dsl import UniversalNeuron
from sc_neurocore.studio.model_catalogue import get_model_detail
from sc_neurocore.studio.model_compile_configuration import resolve_model_compile_configuration
from sc_neurocore.studio.model_numeric_contracts import STUDIO_Q_FORMATS, studio_numeric_contracts


def test_every_candidate_format_has_a_contract_in_order() -> None:
    contracts = studio_numeric_contracts(UniversalNeuron.from_schema("adex"))
    assert tuple(contracts) == STUDIO_Q_FORMATS == ("Q8.8", "Q16.16")
    assert [contract.q_format for contract in contracts.values()] == ["Q8.8", "Q16.16"]
    assert [contract.representable for contract in contracts.values()] == [False, True]


def test_a_model_no_candidate_can_hold_is_offered_no_format() -> None:
    """Its clip bound of 1e6 exceeds even Q16.16, which tops out below 32768."""
    detail = get_model_detail("SCClippedRationalRecoveryMapNeuron")
    assert detail is not None
    configuration = detail["compile_configuration"]
    assert (configuration["q_formats"], configuration["default_q_format"]) == ([], None)
    refusal = configuration["numeric_contracts"]["Q16.16"]["refusal"]
    assert "parameter clip_bound=1000000.0 becomes 16960.0" in refusal
    with pytest.raises(ValueError, match="Q16.16 cannot hold this neuron"):
        resolve_model_compile_configuration(
            {"model_name": "SCClippedRationalRecoveryMapNeuron", "q_format": "Q16.16"}
        )


def test_a_format_studio_does_not_compile_at_is_refused() -> None:
    with pytest.raises(
        ValueError, match=r"Q4\.12 is not one Studio compiles at \(Q8\.8, Q16\.16\)"
    ):
        resolve_model_compile_configuration({"model_name": "AdExNeuron", "q_format": "Q4.12"})


def test_the_compile_refuses_a_format_that_would_wrap_the_model() -> None:
    with pytest.raises(ValueError, match="parameter C=200.0 becomes -56.0"):
        resolve_model_compile_configuration({"model_name": "AdExNeuron", "q_format": "Q8.8"})


def test_parameter_overrides_are_checked_not_the_defaults() -> None:
    """A slider value the format cannot hold refuses the compile at that value."""
    with pytest.raises(ValueError, match="parameter C=40000.0 becomes"):
        resolve_model_compile_configuration(
            {"model_name": "AdExNeuron", "q_format": "Q16.16", "params": {"C": 40000.0}}
        )


def test_the_compile_evidence_carries_the_contract_it_was_built_under() -> None:
    configuration = resolve_model_compile_configuration(
        {"model_name": "AdExNeuron", "q_format": "Q16.16"}
    )
    evidence = configuration.to_public_dict()
    contract = evidence["numeric_contract"]
    assert isinstance(contract, dict)
    assert (contract["q_format"], contract["representable"]) == ("Q16.16", True)
    assert evidence["q_format"] == "Q16.16"
    assert contract == configuration.numeric_contract.to_public_dict()
    capacitance = next(q for q in contract["quantities"] if q["name"] == "C")
    assert (capacitance["rtl_value"], capacitance["status"]) == (200.0, "exact")
