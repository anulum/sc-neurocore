# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Population: vectorized group of identical neurons

"""Population: vectorized group of identical neurons."""

from __future__ import annotations

import numpy as np

from sc_neurocore.neurons import models as _model_registry


def _resolve_model(model):
    """Return a model class from a string name or pass through a class."""
    if isinstance(model, str):
        cls = getattr(_model_registry, model, None)
        if cls is None:
            raise ValueError(f"Unknown model '{model}'. Check neurons.models.__all__.")
        return cls
    return model


class Population:
    """A group of N identical neurons with vectorized state access."""

    def __init__(self, model, n, params=None, label=None):
        """Create *n* neurons of *model* (class or string name)."""
        cls = _resolve_model(model)
        kw = params or {}
        self.neurons = [cls(**kw) for _ in range(n)]
        self.n = n
        self.label = label or cls.__name__
        self._model_cls = cls
        self._voltages = np.zeros(n, dtype=np.float64)
        self._sync_voltages()

    def _sync_voltages(self):
        """Pull membrane voltage from each neuron into the flat array."""
        for i, neuron in enumerate(self.neurons):
            self._voltages[i] = getattr(neuron, "v", 0.0)

    def step_all(self, currents) -> np.ndarray:
        """Advance all neurons one timestep; return binary spike vector."""
        spikes = np.zeros(self.n, dtype=np.int8)
        for i, neuron in enumerate(self.neurons):
            spikes[i] = neuron.step(float(currents[i]))
            self._voltages[i] = getattr(neuron, "v", 0.0)
        return spikes

    def reset_all(self):
        """Reset every neuron to its initial state."""
        for neuron in self.neurons:
            if hasattr(neuron, "reset"):
                neuron.reset()
            elif hasattr(neuron, "reset_state"):
                neuron.reset_state()
        self._sync_voltages()

    def get_states(self) -> dict[str, np.ndarray]:
        """Collect all neuron states into arrays keyed by variable name."""
        if self.n == 0:
            return {}
        sample = self.neurons[0]
        if hasattr(sample, "get_state"):
            keys = sample.get_state().keys()
        elif hasattr(sample, "__dataclass_fields__"):
            keys = [k for k in sample.__dataclass_fields__ if k not in ("dt",)]
        else:
            keys = ["v"]
        result = {}
        for k in keys:
            result[k] = np.array([getattr(n, k, 0.0) for n in self.neurons])
        return result

    @property
    def voltages(self) -> np.ndarray:
        """Current membrane voltages (read-only view)."""
        return self._voltages
