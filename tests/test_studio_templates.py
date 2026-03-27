# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
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


class TestTemplateStructure:
    def test_five_templates_exist(self):
        assert len(TEMPLATES) == 5

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_template_has_required_keys(self, name):
        t = TEMPLATES[name]
        missing = REQUIRED_KEYS - set(t.keys())
        assert not missing, f"{name} missing keys: {missing}"

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_equations_is_nonempty_list(self, name):
        assert isinstance(TEMPLATES[name]["equations"], list)
        assert len(TEMPLATES[name]["equations"]) > 0

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_params_is_dict(self, name):
        assert isinstance(TEMPLATES[name]["params"], dict)

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_init_is_dict(self, name):
        assert isinstance(TEMPLATES[name]["init"], dict)

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_dt_positive(self, name):
        assert TEMPLATES[name]["dt"] > 0

    @pytest.mark.parametrize("name", list(TEMPLATES.keys()))
    def test_duration_positive(self, name):
        assert TEMPLATES[name]["duration"] > 0


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


class TestListAndGet:
    def test_list_returns_all(self):
        result = list_templates()
        assert len(result) == 5
        assert all(isinstance(t, dict) for t in result)

    def test_get_existing(self):
        t = get_template("lif")
        assert t is not None
        assert t["name"] == "lif"

    def test_get_nonexistent(self):
        assert get_template("nonexistent") is None
