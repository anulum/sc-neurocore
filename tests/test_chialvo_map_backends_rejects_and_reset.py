# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_reset) from former test_chialvo_map_backends.py

from __future__ import annotations

from tests.chialvo_map_backends_support import *  # noqa: F403

def test_invalid_backend_rejected() -> None:
    for backend in ("cuda", "", "RUST"):
        with pytest.raises(ValueError, match="backend must be"):
            ChialvoMapNeuron().simulate(1, backend=backend)

def test_invalid_batch_arguments_and_mutable_configuration_rejected() -> None:
    neuron = ChialvoMapNeuron()
    with pytest.raises(ValueError, match="non-negative"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, current=np.inf)
    neuron.k = np.nan
    with pytest.raises(ValueError, match="k"):
        neuron.simulate(1)

def test_reset_preserves_configuration() -> None:
    neuron = ChialvoMapNeuron(a=0.8, b=0.4, c=0.2, k=0.03, x_threshold=0.75)
    neuron.step(0.01)
    neuron.reset()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.a, neuron.b, neuron.c, neuron.k, neuron.x_threshold) == (
        0.8,
        0.4,
        0.2,
        0.03,
        0.75,
    )
