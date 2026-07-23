# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for Studio neuron ODE templates

from __future__ import annotations

import pytest

from sc_neurocore.studio.templates import TEMPLATES, get_template, list_templates

REQUIRED_KEYS = {
    "name",
    "description",
    "equations",
    "threshold",
    "reset",
    "params",
    "init",
    "dt",
    "current",
    "duration",
}

__all__ = [
    "annotations",
    "pytest",
    "TEMPLATES",
    "get_template",
    "list_templates",
    "REQUIRED_KEYS",
]
