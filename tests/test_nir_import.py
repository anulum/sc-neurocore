# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Source/config provenance header

"""Tests for the lightweight dict-form NIR importer.

Covers the broadened node-type support (the six NIR point-neuron types plus the
Izhikevich extension), the shared-template reconciliation with
``nir_bridge.neuron_templates``, alias/fallback resolution, multi-compartment
state equations, and threshold/reset/parameter resolution.
"""

from __future__ import annotations

import pytest

from sc_neurocore.compiler.intelligence.nir_import import (
    NEURON_TEMPLATES,
    import_nir_graph,
)


def _one(node_type=None, **params):
    spec = dict(params)
    if node_type is not None:
        spec["type"] = node_type
    return import_nir_graph({"nodes": {"n0": spec}, "edges": []})


class TestSharedTemplates:
    def test_equations_come_from_the_authoritative_bridge_table(self):
        # The importer must not carry its own divergent LIF dynamics: the leak
        # term and reset come straight from the shared template.
        g = _one("LIF", tau=20)
        assert g.equations["n0"] == "-(v - 0.0) / 20.0 + I * 1.0 / 20.0"
        assert g.thresholds["n0"] == "v > 1.0"
        assert g.resets["n0"] == "v = 0.0"

    def test_bridge_types_are_all_recognised(self):
        for canonical in NEURON_TEMPLATES:
            g = import_nir_graph({"nodes": {"n": {"type": canonical}}, "edges": []})
            assert g.node_types["n"] == canonical


class TestNodeTypes:
    def test_default_type_is_lif(self):
        g = _one()  # no "type" key
        assert g.node_types["n0"] == "lif"

    def test_if_has_threshold_no_leak(self):
        g = _one("IF")
        assert g.thresholds["n0"] == "v > 1.0"
        assert "tau" not in g.equations["n0"]

    def test_li_has_no_threshold(self):
        g = _one("LI", tau=15)
        assert g.thresholds["n0"] is None and g.resets["n0"] is None
        assert "15.0" in g.equations["n0"]

    def test_integrator_pure(self):
        g = _one("I")
        assert g.node_types["n0"] == "integrator"
        assert g.thresholds["n0"] is None

    def test_cuba_lif_is_two_compartment(self):
        g = _one("CubaLIF")
        assert set(g.state_equations["n0"]) == {"i_syn", "v"}
        assert g.thresholds["n0"] == "v > 1.0"
        # membrane equation is the one exposed flat
        assert g.equations["n0"] == g.state_equations["n0"]["v"]

    def test_cuba_li_has_no_threshold(self):
        g = _one("CubaLI")
        assert set(g.state_equations["n0"]) == {"i_syn", "v"}
        assert g.thresholds["n0"] is None


class TestIzhikevichExtension:
    def test_izhikevich_full_model(self):
        g = _one("Izhikevich")
        assert set(g.state_equations["n0"]) == {"u", "v"}
        assert "0.04" in g.equations["n0"]
        assert g.thresholds["n0"] == "v > 30"
        assert g.resets["n0"] == "v = -65.0; u = u + 8.0"

    def test_izhikevich_params_override(self):
        g = _one("izh", a=0.1, b=0.25)
        assert g.parameters["n0"]["a"] == 0.1 and g.parameters["n0"]["b"] == 0.25
        assert "u = u + 8.0" in g.resets["n0"]


class TestAliasesAndFallback:
    @pytest.mark.parametrize(
        "alias,canonical",
        [
            ("LeakyIntegrateAndFire", "lif"),
            ("integrate_and_fire", "if"),
            ("Leaky Integrator", "li"),
            ("cuba-lif", "cuba_lif"),
            ("CUBALI", "cuba_li"),
            ("Integrator", "integrator"),
        ],
    )
    def test_alias_resolution(self, alias, canonical):
        assert _one(alias).node_types["n0"] == canonical

    def test_unknown_type_falls_back_to_leaky_integrator(self):
        g = _one("Mystery", tau=5.0)
        assert g.node_types["n0"] == "li"
        assert "5.0" in g.equations["n0"]


class TestParameterResolution:
    def test_defaults_are_applied(self):
        g = _one("LIF")
        assert g.parameters["n0"]["tau"] == 20.0
        assert g.parameters["n0"]["v_threshold"] == 1.0

    def test_node_params_override_defaults(self):
        g = _one("LIF", tau=7.0, v_threshold=2.0)
        assert g.parameters["n0"]["tau"] == 7.0
        assert "7.0" in g.equations["n0"]
        assert g.thresholds["n0"] == "v > 2.0"

    def test_unknown_params_are_ignored(self):
        g = _one("LIF", not_a_param=99.0)
        assert "not_a_param" not in g.parameters["n0"]

    def test_distinct_time_constants_substituted_independently(self):
        # tau_syn and tau_mem must not clobber one another (longest-first).
        g = _one("CubaLIF", tau_syn=3.0, tau_mem=17.0)
        assert "3.0" in g.state_equations["n0"]["i_syn"]
        assert "17.0" in g.state_equations["n0"]["v"]
        assert "tau" not in g.state_equations["n0"]["v"]


class TestGraphStructure:
    def test_edges_and_framework(self):
        g = import_nir_graph(
            {"nodes": {"a": {"type": "LIF"}, "b": {"type": "LI"}}, "edges": [["a", "b"]]},
            framework="Norse",
        )
        assert ("a", "b") in g.edges
        assert g.framework == "Norse"
        assert set(g.node_types) == {"a", "b"}

    def test_empty_graph(self):
        g = import_nir_graph({})
        assert g.equations == {} and g.edges == []
