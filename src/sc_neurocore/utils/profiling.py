# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Memory estimation for SC-NeuroCore networks

"""Memory estimation for SC-NeuroCore networks."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_memory(layers: list[Any], unit: str = "MB") -> dict[str, Any]:
    """Estimate memory usage of a list of SC layers.

    Example::

        from sc_neurocore import VectorizedSCLayer
        from sc_neurocore.utils.profiling import estimate_memory

        layers = [
            VectorizedSCLayer(n_inputs=50, n_neurons=128, length=256),
            VectorizedSCLayer(n_inputs=128, n_neurons=10, length=256),
        ]
        print(estimate_memory(layers))
        # {'weights_bytes': 56320, 'packed_bytes': 112640, ...}

    Parameters
    ----------
    layers : list
        SC layer objects with ``.weights`` and ``.length`` attributes.
    unit : str
        "B", "KB", or "MB".

    Returns
    -------
    dict
        Breakdown: weights_bytes, packed_bytes, neuron_state_bytes,
        total_bytes, total_human.
    """
    divisors = {"B": 1, "KB": 1024, "MB": 1024**2}
    div = divisors.get(unit, 1)

    weights_bytes = 0
    packed_bytes = 0
    neuron_state_bytes = 0

    for layer in layers:
        w = getattr(layer, "weights", None)
        if w is not None:
            weights_bytes += w.nbytes

        L = getattr(layer, "length", 256)
        if w is not None:
            n_out, n_in = w.shape
            # Packed bitstreams: each weight is L bits packed into uint64 words
            words_per_weight = int(np.ceil(L / 64))
            packed_bytes += n_out * n_in * words_per_weight * 8

            # Neuron state: voltage (float64) + spike flag per neuron
            neuron_state_bytes += n_out * 9  # 8 bytes float + 1 byte flag

    total = weights_bytes + packed_bytes + neuron_state_bytes

    return {
        "weights_bytes": weights_bytes,
        "packed_bytes": packed_bytes,
        "neuron_state_bytes": neuron_state_bytes,
        "total_bytes": total,
        "total_human": f"{total / div:.2f} {unit}",
    }
