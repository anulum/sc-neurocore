# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Studio network canvas creation

"""Focused suite: TestCreation from former test_studio_network_canvas.py."""

from __future__ import annotations

import pytest
from tests.studio_network_canvas_support import (
    available_models,
    create_population,
    create_projection,
)
from typing import Never


class TestCreation:
    """Creating populations and projections for the network canvas."""

    def test_create_population_defaults(self) -> None:
        """A default population is 80 excitatory LIF neurons without parameters or drive."""
        pop = create_population()
        assert pop["type"] == "population"
        assert pop["id"].startswith("pop_")
        assert pop["count"] == 80
        assert pop["neuron_type"] == "excitatory"
        assert pop["model"] == "SCLapicqueLIFNeuron"
        assert pop["params"] == {}
        assert pop["drive"] == {"kind": "none"}

    def test_create_population_carries_params_and_drive(self) -> None:
        """A population keeps its parameters and drive."""
        pop = create_population(params={"tau": 10.0}, drive={"kind": "constant", "current": 1.2})
        assert pop["params"] == {"tau": 10.0}
        assert pop["drive"] == {"kind": "constant", "current": 1.2}

    def test_create_population_inhibitory(self) -> None:
        """An inhibitory population keeps its count and label."""
        pop = create_population(
            label="Inh 0", model="LIFNeuron", count=20, neuron_type="inhibitory"
        )
        assert pop["neuron_type"] == "inhibitory"
        assert pop["count"] == 20
        assert pop["label"] == "Inh 0"

    def test_create_projection(self) -> None:
        """A random projection keeps its endpoints, weight, delay and probability."""
        proj = create_projection("src", "tgt", weight=0.3, delay=2.0, probability=0.5)
        assert proj["id"].startswith("proj_")
        assert proj["source"] == "src"
        assert proj["target"] == "tgt"
        assert proj["weight"] == 0.3
        assert proj["delay"] == 2.0
        assert proj["rule"] == "random"
        assert proj["probability"] == 0.5

    def test_create_projection_all_to_all_carries_no_probability(self) -> None:
        """An all-to-all projection carries no probability and a zero delay."""
        proj = create_projection("src", "tgt", weight=-2.0, rule="all_to_all")
        assert proj["rule"] == "all_to_all"
        assert "probability" not in proj
        assert proj["delay"] == 0.0

    def test_unique_ids(self) -> None:
        """Ten populations get ten distinct ids."""
        ids = {create_population()["id"] for _ in range(10)}
        assert len(ids) == 10

    def test_available_models(self) -> None:
        """More than 100 models are available, AdEx among them."""
        models = available_models()
        assert isinstance(models, list)
        assert len(models) > 100
        assert "AdExNeuron" in models

    def test_available_models_raises_on_catalog_failure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A catalogue failure raises RuntimeError."""
        import sc_neurocore.studio.network_graph as mod

        def _boom() -> Never:
            raise RuntimeError("catalog failed")

        monkeypatch.setattr(mod, "list_models", _boom)
        with pytest.raises(RuntimeError, match="catalog failed"):
            mod.available_models()
