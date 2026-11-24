# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio templates template structure

"""Focused suite: TestTemplateStructure from former test_studio_templates.py."""

from __future__ import annotations

import pytest
from sc_neurocore.studio.templates import (
    TEMPLATES,
)
from tests.studio_templates_support import (
    REQUIRED_KEYS,
)


class TestTemplateStructure:
    """The structure of the Studio templates."""

    def test_five_templates_exist(self) -> None:
        """There are five templates."""
        assert len(TEMPLATES) == 5

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_template_has_required_keys(self, name: str) -> None:
        """Each template has every required key."""
        t = TEMPLATES[name]
        missing = REQUIRED_KEYS - set(t.keys())
        assert not missing, f"{name} missing keys: {missing}"

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_equations_is_nonempty_list(self, name: str) -> None:
        """Each template's equations are a non-empty list."""
        assert isinstance(TEMPLATES[name]["equations"], list)
        assert len(TEMPLATES[name]["equations"]) > 0

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_params_is_dict(self, name: str) -> None:
        """Each template's parameters are a dictionary."""
        assert isinstance(TEMPLATES[name]["params"], dict)

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_init_is_dict(self, name: str) -> None:
        """Each template's initial state is a dictionary."""
        assert isinstance(TEMPLATES[name]["init"], dict)

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_dt_positive(self, name: str) -> None:
        """Each template's time step is positive."""
        assert TEMPLATES[name]["dt"] > 0

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_duration_positive(self, name: str) -> None:
        """Each template's duration is positive."""
        assert TEMPLATES[name]["duration"] > 0
