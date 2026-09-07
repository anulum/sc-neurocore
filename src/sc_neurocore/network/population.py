# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Population: vectorized group of identical neurons

"""Population: vectorized group of identical neurons."""

from __future__ import annotations

import copy
import inspect
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from sc_neurocore.network.population_seeds import derive_population_seeds
from sc_neurocore.neurons.seed_domain import seed_domain
from sc_neurocore.network.quiescence import (
    StateSignature,
    is_quiescent,
    quiescent_signature,
)
from sc_neurocore.neurons import models as _model_registry


def _resolve_model(model: type[Any] | str) -> type[Any]:
    """Return a model class from a string name or pass through a class."""
    if isinstance(model, str):
        cls: type[Any] | None = getattr(_model_registry, model, None)
        if cls is None:
            raise ValueError(f"Unknown model '{model}'. Check neurons.models.__all__.")
        return cls
    return model


def _seed_default(factory: object) -> int | None:
    """Return the seed a factory would use when the caller names none.

    Parameters
    ----------
    factory : object
        The constructor a population's neurons are built with. Typed as
        ``object`` rather than as a callable because the answer for something
        that cannot be asked — a builtin with no signature, or a value that is
        not callable at all — is part of this function's contract.

    Returns
    -------
    int or None
        The default seed, or ``None`` when the factory takes no ``seed``
        parameter or defaults it to independent entropy. ``None`` is returned
        unchanged because a model that defaults to entropy already gives every
        neuron its own stream.
    """
    if not callable(factory):
        # Not every value handed to a population is a constructor; one that is
        # not callable declares nothing, and is given no seed.
        return None
    try:
        parameter = inspect.signature(factory).parameters.get("seed")
    except ValueError:
        # A C-implemented factory need not carry a signature at all: `inspect`
        # raises ValueError for a builtin type. It cannot be asked about a
        # seed, so it is not given one. Exercised rather than assumed.
        return None
    if parameter is None or parameter.default is inspect.Parameter.empty:
        return None
    default = parameter.default
    return default if isinstance(default, int) and not isinstance(default, bool) else None


def _per_neuron_kwargs(
    factory: Callable[..., Any], kw: dict[str, Any], n: int
) -> Iterator[dict[str, Any]]:
    """Yield the constructor arguments for each neuron of a population.

    Every neuron of a seeded model used to receive the same seed, so a
    population of stochastic neurons produced one spike train repeated ``n``
    times. Each neuron is given its own derived seed instead, distinct and
    reproducible from the population's base seed.

    An explicit ``seed=None`` is passed through unchanged: it already asks each
    model for independent entropy, and deriving over it would replace the
    caller's request for unpredictability with a reproducible sequence.

    Parameters
    ----------
    factory : callable
        The constructor the neurons are built with.
    kw : dict
        The parameters every neuron shares.
    n : int
        How many neurons the population holds.

    Yields
    ------
    dict
        One neuron's constructor arguments.
    """
    base_seed = kw["seed"] if "seed" in kw else _seed_default(factory)
    if base_seed is None or isinstance(base_seed, bool) or not isinstance(base_seed, int):
        yield from ({**kw} for _ in range(n))
        return
    # Derive inside the domain the model declares, not the narrowest one any
    # model has: a model with a 63-bit domain was receiving 16-bit seeds only
    # because nothing published what it accepts.
    domain = seed_domain(factory if isinstance(factory, type) else type(factory))
    for seed in derive_population_seeds(base_seed, n, domain):
        yield {**kw, "seed": seed}


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
        factory: Callable[..., Any] = cls
        if isinstance(model, str) and model == "LapicqueNeuron":
            source_fields = {
                "v",
                "v_threshold",
                "capacitance",
                "series_resistance",
                "polarization_resistance",
                "dt",
                "excited",
            }
            sc_only_fields = {"v_rest", "v_reset", "tau", "resistance"}
            if not (set(kw) & sc_only_fields):
                if not set(kw) <= source_fields:
                    unexpected = sorted(set(kw) - source_fields)
                    raise TypeError(
                        "LapicqueNeuron source profile received unsupported parameters: "
                        + ", ".join(unexpected)
                    )
                factory = cls.lapicque_1907
        self.neurons = [factory(**kwargs) for kwargs in _per_neuron_kwargs(factory, kw, n)]
        self.n = n
        self.model_name = cls.__name__
        self.label = label or cls.__name__
        self._model_cls = cls
        self._voltages = np.zeros(n, dtype=np.float64)
        self._quiescent: StateSignature | None = None
        self._quiescent_probed = False
        self._sync_voltages()

    def _sync_voltages(self) -> None:
        """Pull membrane voltage from each neuron into the flat array."""
        for i, neuron in enumerate(self.neurons):
            self._voltages[i] = getattr(neuron, "v", 0.0)

    def quiescent_signature(self) -> StateSignature | None:
        """Return the state this population's model holds under zero input.

        Measured once and cached. The probe runs on a reset copy of this
        population's own first neuron, not on a default instance, so a
        population built with parameters is measured at *its* rest state rather
        than the model's. ``None`` when the model does not hold still — a
        pacemaker, an oscillator, anything that moves or spikes without input —
        in which case no neuron of this population is ever skipped.
        """
        if self._quiescent_probed:
            return self._quiescent
        self._quiescent_probed = True
        if not self.neurons:
            return self._quiescent
        try:
            resting = copy.deepcopy(self.neurons[0])
        except Exception:
            return self._quiescent
        for name in ("reset", "reset_state"):
            method = getattr(resting, name, None)
            if callable(method):
                try:
                    method()
                except Exception:
                    return self._quiescent
                break
        self._quiescent = quiescent_signature(resting)
        return self._quiescent

    def step_all(
        self, currents: np.ndarray[Any, Any], spike_gating: bool = False
    ) -> np.ndarray[Any, Any]:
        """Advance all neurons one timestep; return binary spike vector.

        If *spike_gating* is True, a neuron is skipped only when skipping it
        and stepping it are the same thing: its input is exactly zero and its
        whole state matches a state this model's zero-input map was measured to
        leave unchanged. Compute then falls towards the active fraction of a
        sparse network without the run differing from the same run ungated.

        The test is exact rather than a tolerance around rest. A neuron a
        little away from rest is precisely the one whose relaxation a skip
        would discard, and skipping it froze a leak, an adaptation current or a
        refractory countdown that the model would have advanced.
        """
        spikes = np.zeros(self.n, dtype=np.int8)
        if spike_gating:
            signature = self.quiescent_signature()
            for i, neuron in enumerate(self.neurons):
                if currents[i] == 0.0 and is_quiescent(neuron, signature):
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
