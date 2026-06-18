# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Population: vectorized group of identical neurons

"""Population: vectorized group of identical neurons."""

from __future__ import annotations

from typing import Any

import numpy as np

from sc_neurocore.neurons import models as _model_registry


def _resolve_model(model: type[Any] | str) -> type[Any]:
    """Return a model class from a string name or pass through a class."""
    if isinstance(model, str):
        cls: type[Any] | None = getattr(_model_registry, model, None)
        if cls is None:
            raise ValueError(f"Unknown model '{model}'. Check neurons.models.__all__.")
        return cls
    return model


class Population:
    """A group of N identical neurons with vectorized state access."""

    def __init__(
        self,
        model: type[Any] | str,
        n: int,
        params: dict[str, Any] | None = None,
        label: str | None = None,
    ) -> None:
        """Create *n* neurons of *model* (class or string name)."""
        cls = _resolve_model(model)
        kw = params or {}
        self.neurons = [cls(**kw) for _ in range(n)]
        self.n = n
        self.model_name = cls.__name__
        self.label = label or cls.__name__
        self._model_cls = cls
        self._voltages = np.zeros(n, dtype=np.float64)
        self._sync_voltages()

    def _sync_voltages(self) -> None:
        """Pull membrane voltage from each neuron into the flat array."""
        for i, neuron in enumerate(self.neurons):
            self._voltages[i] = getattr(neuron, "v", 0.0)

    def step_all(
        self, currents: np.ndarray[Any, Any], spike_gating: bool = False
    ) -> np.ndarray[Any, Any]:
        """Advance all neurons one timestep; return binary spike vector.

        If *spike_gating* is True, neurons with zero input current and
        voltage near rest (within 1% of threshold) are skipped. This
        makes compute roughly proportional to active neurons — useful for
        sparse-firing networks. Skipped neurons still decay via leak if
        their model tracks sub-threshold dynamics.
        """
        spikes = np.zeros(self.n, dtype=np.int8)
        if spike_gating:
            for i, neuron in enumerate(self.neurons):
                v = getattr(neuron, "v", 0.0)
                v_thresh = getattr(neuron, "v_threshold", 1.0)
                v_rest = getattr(neuron, "v_rest", 0.0)
                # Skip if no input AND voltage within 1% of rest
                if currents[i] == 0.0 and abs(v - v_rest) < 0.01 * abs(v_thresh - v_rest):
                    continue
                raw = neuron.step(float(currents[i]))
                spikes[i] = min(max(int(raw), 0), 1)
                self._voltages[i] = getattr(neuron, "v", 0.0)
        else:
            for i, neuron in enumerate(self.neurons):
                raw = neuron.step(float(currents[i]))
                spikes[i] = min(max(int(raw), 0), 1)
                self._voltages[i] = getattr(neuron, "v", 0.0)
        return spikes

    def reset_all(self) -> None:
        """Reset every neuron to its initial state."""
        for neuron in self.neurons:
            if hasattr(neuron, "reset"):
                neuron.reset()
            elif hasattr(neuron, "reset_state"):
                neuron.reset_state()
        self._sync_voltages()

    def get_states(self) -> dict[str, np.ndarray[Any, Any]]:
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

    def set_voltages(self, voltages: np.ndarray[Any, Any]) -> None:
        """Sync voltages from an external source (e.g. Rust backend) into neurons."""
        for i, neuron in enumerate(self.neurons):
            if hasattr(neuron, "v"):
                neuron.v = float(voltages[i])
        self._voltages[:] = voltages[: self.n]

    @property
    def voltages(self) -> np.ndarray[Any, Any]:
        """Current membrane voltages (read-only view)."""
        return self._voltages
