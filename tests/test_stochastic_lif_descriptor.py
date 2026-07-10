# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — StochasticLIF catalogue registration tests

"""Real-surface tests: public StochasticLIF is registered and describable."""

from __future__ import annotations

from sc_neurocore import StochasticLIFNeuron as PublicStochasticLIF
from sc_neurocore.neurons.descriptor_tiers import science_tier, silicon_tier
from sc_neurocore.neurons.model_catalogue import (
    catalogue_descriptor_coverage,
    load_descriptor,
)
from sc_neurocore.neurons.models import StochasticLIFNeuron as ModelsStochasticLIF
from sc_neurocore.neurons.models import _CLASS_TO_MODULE
from sc_neurocore.studio.models import get_model_detail, list_models


def test_stochastic_lif_registered_and_same_class() -> None:
    """Package root and models registry expose the same implementation class."""
    assert _CLASS_TO_MODULE["StochasticLIFNeuron"] == "stochastic_lif"
    assert ModelsStochasticLIF is PublicStochasticLIF
    neuron = PublicStochasticLIF(v_threshold=1.0, tau_mem=20.0, noise_std=0.0)
    spikes = sum(neuron.step(0.8) for _ in range(200))
    assert spikes >= 0  # finite run; non-negative spike count


def test_stochastic_lif_descriptor_loads_with_dynamics() -> None:
    """Committed descriptor is valid and carries dynamics for S4 readiness path."""
    desc = load_descriptor("StochasticLIFNeuron")
    assert desc is not None
    assert desc.class_name == "StochasticLIFNeuron"
    assert desc.module == "stochastic_lif"
    assert desc.dynamics
    assert desc.provenance.is_citeable
    assert all(p.is_curated for p in desc.parameters)
    # Curation kernel at least S2; silicon not claimed (no schema cosim for this class).
    assert science_tier(desc) >= 2
    assert silicon_tier(desc) is None


def test_catalogue_coverage_includes_stochastic_lif() -> None:
    """Coverage total counts the new registry entry."""
    cov = catalogue_descriptor_coverage()
    assert cov.total_models == len(_CLASS_TO_MODULE)
    assert cov.described >= 1
    assert load_descriptor("StochasticLIFNeuron") is not None


def test_studio_list_models_includes_stochastic_lif() -> None:
    """Studio catalogue surfaces the public SC flagship."""
    # Clear studio cache if present.
    import sc_neurocore.studio.models as studio_models

    studio_models._models_cache = None
    names = {m["name"] for m in list_models()}
    assert "StochasticLIFNeuron" in names
    detail = get_model_detail("StochasticLIFNeuron")
    assert detail is not None
    assert detail["name"] == "StochasticLIFNeuron"
    assert detail["science_tier"] >= 2
