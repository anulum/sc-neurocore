# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas creation

"""Focused suite: TestCreation from former test_studio_network_canvas.py."""

from __future__ import annotations

from tests.studio_network_canvas_support import *  # noqa: F403

class TestCreation:
    def test_create_population_defaults(self):
        pop = create_population()
        assert pop["type"] == "population"
        assert pop["id"].startswith("pop_")
        assert pop["count"] == 80
        assert pop["neuron_type"] == "excitatory"

    def test_create_population_inhibitory(self):
        pop = create_population(
            label="Inh 0", model="LIFNeuron", count=20, neuron_type="inhibitory"
        )
        assert pop["neuron_type"] == "inhibitory"
        assert pop["count"] == 20
        assert pop["label"] == "Inh 0"

    def test_create_projection(self):
        proj = create_projection("src", "tgt", weight=0.3, delay=2.0, probability=0.5)
        assert proj["id"].startswith("proj_")
        assert proj["source"] == "src"
        assert proj["target"] == "tgt"
        assert proj["weight"] == 0.3
        assert proj["delay"] == 2.0

    def test_unique_ids(self):
        ids = {create_population()["id"] for _ in range(10)}
        assert len(ids) == 10

    def test_available_models(self):
        models = available_models()
        assert isinstance(models, list)
        assert len(models) > 100
        assert "AdExNeuron" in models

    def test_available_models_raises_on_catalog_failure(self, monkeypatch):
        import sc_neurocore.studio.network_graph as mod

        def _boom():
            raise RuntimeError("catalog failed")

        monkeypatch.setattr(mod, "list_models", _boom)
        with pytest.raises(RuntimeError, match="catalog failed"):
            mod.available_models()

