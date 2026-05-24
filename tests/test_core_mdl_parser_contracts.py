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
