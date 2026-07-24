# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFormatParity from former test_universal_dsl.py

"""Focused suite: TestFormatParity from former test_universal_dsl.py."""

from __future__ import annotations

from tests.universal_dsl_support import *  # noqa: F403


class TestFormatParity:
    """Verify that TOML and JSON schemas produce identical simulation results."""

    def test_lif_toml_vs_json_parity(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"

        toml_neuron = UniversalNeuron.from_schema(schema_dir / "lif.toml")
        json_neuron = UniversalNeuron.from_schema(schema_dir / "lif.json")

        for _ in range(100):
            s1 = toml_neuron.step(I=20.0)
            s2 = json_neuron.step(I=20.0)
            assert s1 == s2

        assert toml_neuron.state["v"] == json_neuron.state["v"]

    def test_izhikevich_toml_vs_json_parity(self) -> None:
        schema_dir = Path(__file__).parent.parent / "src/sc_neurocore/neurons/model_schemas"

        toml_neuron = UniversalNeuron.from_schema(schema_dir / "izhikevich.toml")
        json_neuron = UniversalNeuron.from_schema(schema_dir / "izhikevich.json")

        for _ in range(100):
            s1 = toml_neuron.step(I=10.0)
            s2 = json_neuron.step(I=10.0)
            assert s1 == s2

        for var in ("v", "u"):
            assert toml_neuron.state[var] == json_neuron.state[var]
