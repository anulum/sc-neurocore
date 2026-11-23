# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestDocGenerator from former test_model_zoo.py

"""Focused suite: TestDocGenerator from former test_model_zoo.py."""

from __future__ import annotations

from sc_neurocore.model_zoo.model_zoo import (
    DocGenerator,
    IzhikevichPlugin,
    LIFPlugin,
    PluginRegistry,
)


class TestDocGenerator:
    """Markdown documentation generated for model plugins."""

    def test_generates_markdown(self) -> None:
        """A plugin page has its title, parameters and default values."""
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert md.startswith("# LIF")
        assert "## Parameters" in md
        assert "## Default Values" in md

    def test_references_section(self) -> None:
        """The LIF page lists its references, Lapicque among them."""
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "## References" in md
        assert "Lapicque" in md

    def test_state_variables_listed(self) -> None:
        """The Izhikevich page lists its state variables."""
        doc = DocGenerator()
        md = doc.generate(IzhikevichPlugin())
        assert "## State Variables" in md
        assert "`V`" in md
        assert "`u`" in md

    def test_index_generation(self) -> None:
        """The model-zoo index names every built-in model."""
        reg = PluginRegistry.with_builtins()
        doc = DocGenerator()
        index = doc.generate_index(reg)
        assert "# SC-NeuroCore Model Zoo" in index
        assert "LIF" in index
        assert "Izhikevich" in index
        assert "AdEx" in index
        assert "Hodgkin-Huxley" in index

    def test_default_values_table(self) -> None:
        """The LIF page shows the membrane time-constant default."""
        doc = DocGenerator()
        md = doc.generate(LIFPlugin())
        assert "0.02" in md  # tau_m default
