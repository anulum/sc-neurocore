# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDocGenerator from former test_model_zoo.py

"""Focused suite: TestDocGenerator from former test_model_zoo.py."""

from __future__ import annotations

from model_zoo_support import *  # noqa: F403


class TestDocGenerator:
    def test_generates_markdown(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert md.startswith("# LIF")
        assert "## Parameters" in md
        assert "## Default Values" in md

    def test_references_section(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "## References" in md
        assert "Lapicque" in md

    def test_state_variables_listed(self):
        doc = DocGenerator()
        md = doc.generate(IzhikevichPlugin())
        assert "## State Variables" in md
        assert "`V`" in md
        assert "`u`" in md

    def test_index_generation(self):
        reg = PluginRegistry.with_builtins()
        doc = DocGenerator()
        index = doc.generate_index(reg)
        assert "# SC-NeuroCore Model Zoo" in index
        assert "LIF" in index
        assert "Izhikevich" in index
        assert "AdEx" in index
        assert "Hodgkin-Huxley" in index

    def test_default_values_table(self):
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "0.02" in md  # tau_m default
