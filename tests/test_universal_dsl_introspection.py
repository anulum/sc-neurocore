# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestIntrospection from former test_universal_dsl.py

"""Focused suite: TestIntrospection from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403

class TestIntrospection:
    """Test introspection methods."""

    def test_name(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        assert neuron.name == "FitzHugh-Nagumo"

    def test_doi(self) -> None:
        neuron = UniversalNeuron.from_schema("izhikevich")
        assert neuron.doi == "10.1109/TNN.2003.820440"

    def test_list_state_variables(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        assert set(neuron.list_state_variables()) == {"x", "y", "z"}

    def test_list_parameters(self) -> None:
        neuron = UniversalNeuron.from_schema("hindmarsh_rose")
        params = neuron.list_parameters()
        assert "b" in params
        assert "r" in params
        assert "s" in params

    def test_list_equations(self) -> None:
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")
        eqs = neuron.list_equations()
        assert "v" in eqs
        assert "w" in eqs
        # ``v * v * v`` (exact IEEE cube), not ``v ** 3`` — matches the hand model's RHS.
        assert "v * v * v" in eqs["v"]

    def test_repr(self) -> None:
        neuron = UniversalNeuron.from_schema("lif")
        r = repr(neuron)
        assert "UniversalNeuron" in r
        assert "LIF" in r

    def test_extensions_property(self) -> None:
        neuron = UniversalNeuron.from_schema("adex")
        ext = neuron.extensions
        assert "integrator_options" in ext

    def test_to_equation_neuron(self) -> None:
        from sc_neurocore.neurons.equation_builder import EquationNeuron

        neuron = UniversalNeuron.from_schema("lif")
        eq_neuron = neuron.to_equation_neuron()
        assert isinstance(eq_neuron, EquationNeuron)

    def test_to_verilog_sanitizes_default_module_name(
        self,
        monkeypatch: MonkeyPatch,
    ) -> None:
        captured: dict[str, object] = {}

        def fake_compile_to_verilog(
            neuron: object,
            *,
            module_name: str,
            **kwargs: Any,
        ) -> str:
            captured["neuron"] = neuron
            captured["module_name"] = module_name
            captured["kwargs"] = kwargs
            return f"module {module_name}; endmodule"

        monkeypatch.setattr(
            "sc_neurocore.compiler.equation_compiler.compile_to_verilog",
            fake_compile_to_verilog,
        )
        neuron = UniversalNeuron.from_schema("fitzhugh_nagumo")

        verilog = neuron.to_verilog(data_width=12)

        assert verilog == "module sc_fitzhugh_nagumo; endmodule"
        assert captured["neuron"] is neuron.to_equation_neuron()
        assert captured["module_name"] == "sc_fitzhugh_nagumo"
        assert captured["kwargs"] == {"data_width": 12}
