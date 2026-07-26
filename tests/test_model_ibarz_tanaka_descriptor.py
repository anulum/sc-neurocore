# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka descriptor and default tests

"""Published defaults and descriptor topology for the Ibarz-Tanaka map."""

from __future__ import annotations

import inspect

from sc_neurocore.neurons.model_catalogue import load_descriptor_payload
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def test_defaults_match_the_published_figure_protocol() -> None:
    """Defaults use the paper's alpha, mu, sigma and Fig. 2 map placement."""
    neuron = IbarzTanakaMapNeuron()
    assert (neuron.v, neuron.u) == (-1.0, -0.1)
    assert (neuron.alpha, neuron.mu, neuron.sigma) == (1.0, 0.001, 0.1)


def test_descriptor_structure_matches_the_discrete_map() -> None:
    """Only source parameters are exposed and dt remains integration metadata."""
    payload = load_descriptor_payload("IbarzTanakaMapNeuron")
    assert payload is not None
    assert "dt" not in inspect.signature(IbarzTanakaMapNeuron).parameters
    assert set(payload["state"]) == {"v", "u"}
    assert set(payload["parameters"]) == {"alpha", "mu", "sigma"}
    assert payload["integration"] == {"dt": 1.0, "method": "map"}
