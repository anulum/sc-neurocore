# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio templates template parsing

"""Focused suite: TestTemplateParsing from former test_studio_templates.py."""

from __future__ import annotations

from tests.studio_templates_support import *  # noqa: F403

class TestTemplateParsing:
    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_template_creates_valid_neuron(self, name):
        from sc_neurocore.neurons.equation_builder import from_equations

        t = TEMPLATES[name]
        neuron = from_equations(
            *t["equations"],
            threshold=t["threshold"] if t["threshold"] else None,
            reset=t["reset"] if t["reset"] else None,
            params=t["params"],
            init=t["init"],
            dt=t["dt"],
        )
        assert neuron is not None
        assert len(neuron.state) == len(t["init"])

