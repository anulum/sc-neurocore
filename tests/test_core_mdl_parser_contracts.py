# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for MDL parser contracts

"""Contracts for Mind Description Language module serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from sc_neurocore.core.mdl_parser import MindDescriptionLanguage


def test_mdl_parser_serialises_module_state() -> None:
    class Orchestrator:
        modules = {
            "m": type("M", (), {"get_state": lambda self: {"v": 0.5}, "__module__": __name__})()
        }

    yaml_str = MindDescriptionLanguage.encode(Orchestrator(), "agent")

    assert "m" in yaml_str


def test_mdl_parser_serialises_module_weights() -> None:
    class Orchestrator:
        modules = {"w": type("W", (), {"weights": np.array([0.5]), "__module__": __name__})()}

    yaml_str = MindDescriptionLanguage.encode(Orchestrator(), "agent")

    assert "w" in yaml_str


def test_mdl_parser_decode_round_trips_an_encoded_mind() -> None:
    """decode() parses an encoded MDL string back into its agent mapping."""

    class Orchestrator:
        modules = {
            "m": type("M", (), {"get_state": lambda self: {"v": 0.5}, "__module__": __name__})()
        }

    yaml_str = MindDescriptionLanguage.encode(Orchestrator(), "agent")
    decoded = MindDescriptionLanguage.decode(yaml_str)

    assert isinstance(decoded, dict)
    assert decoded["agent_name"] == "agent"


def test_mdl_parser_decode_rejects_non_mapping() -> None:
    """decode() rejects an MDL string that does not parse into a mapping."""

    with pytest.raises(ValueError, match="mapping"):
        MindDescriptionLanguage.decode("- alpha\n- beta\n")
