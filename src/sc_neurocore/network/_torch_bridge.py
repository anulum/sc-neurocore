# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Explicit torch bridge for declarative Network graphs

"""Torch bridge for declarative ``Network`` graphs.

This module adds a differentiable execution lane without altering the existing
NumPy or Rust inference paths. The bridge is explicit, opt-in, and currently
supports only a bounded subset of neuron models whose state updates can be
mapped without hidden approximations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sc_neurocore.training.surrogate import atan_surrogate_custom_op

from .population import Population
from .projection import Projection, validate_csr_topology


if TYPE_CHECKING:

    class _TorchModuleBase:
        """Typed surface used when optional Torch imports are unavailable to mypy."""

        def __init__(self) -> None: ...

        def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

        def register_buffer(
            self,
            name: str,
            tensor: torch.Tensor | None,
            persistent: bool = True,
        ) -> None: ...

        def register_parameter(
            self,
            name: str,
            param: torch.nn.Parameter | None,
        ) -> None: ...

        def parameters(self, recurse: bool = True) -> Any: ...

else:
    _TorchModuleBase = nn.Module


@dataclass(frozen=True)
class _PopulationSpec:
    population: Population
    state_kind: str
    init_voltage: tuple[float, ...]
    label: str


class _ExactResetCell(nn.Module):
    """The deterministic discrete update of a supported model, with a surrogate gradient.

    The candidate voltage is computed with the model's own operations in the
    model's own order, so a float64 run reproduces the Python neuron bit for bit
    and a factor such as ``R·dt`` is never formed where the model does not form
    it (``R·dt`` can overflow where ``(R·I)·dt`` does not):

    * ``"flow"`` (SC Lapicque): ``v_inf = v_rest + R·I``; ``v_inf + (v - v_inf)·decay``;
    * ``"euler"`` (StochasticLIF): ``v + (-(v - v_rest)·(dt/tau) + (R·I)·dt)``.

    The spike is ``candidate >= threshold`` in the forward pass (the models'
    comparison) with the surrogate's gradient in the backward pass; a spike
    resets to ``v_reset``.
    """

    def __init__(
        self,
        *,
        form: str,
        decay: float,
        resistance: float,
        dt: float,
        v_rest: float,
        v_reset: float,
        threshold: float,
        surrogate_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> None:
        super().__init__()
        self.form = form
        self.decay = decay
        self.resistance = resistance
        self.dt = dt
        self.v_rest = v_rest
        self.v_reset = v_reset
        self.threshold = threshold
        self.surrogate_fn = surrogate_fn

    def forward(self, current: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the spike and the next voltage of one step."""
        if self.form == "flow":
            v_inf = self.v_rest + self.resistance * current
            candidate = v_inf + (v - v_inf) * self.decay
        else:
            leak = -(v - self.v_rest) * self.decay
            candidate = v + (leak + (self.resistance * current) * self.dt)
        distance = candidate - self.threshold
        surrogate = self.surrogate_fn(distance).to(distance.dtype)
        exact = (distance >= 0).to(distance.dtype)
        spike = exact + (surrogate - surrogate.detach())
        # Select rather than blend: a blend multiplies an overflowed candidate by
        # zero on a spike, and inf * 0 would turn the hard reset into NaN.
        next_v = torch.where(exact > 0, torch.full_like(candidate, self.v_reset), candidate)
        return spike, next_v


def _csr_to_dense(projection: Projection) -> np.ndarray[Any, Any]:
    # float64, the CSR's own precision: an untrained sync and a double bridge keep
    # every weight exactly; the forward pass casts to the input's dtype.
    dense = np.zeros((projection.target.n, projection.source.n), dtype=np.float64)
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            dense[tgt_idx, src_idx] = float(projection.data[k])
    return dense


def _csr_mask(projection: Projection) -> np.ndarray[Any, Any]:
    mask = np.zeros((projection.target.n, projection.source.n), dtype=np.float64)
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            mask[tgt_idx, src_idx] = 1.0
    return mask


def _validate_projection_csr(projection: Projection) -> None:
    validate_csr_topology(
        projection.indptr,
        projection.indices,
        projection.data,
        projection.source.n,
        projection.target.n,
        context="Network.to_torch() projection",
    )
    # One dense weight per (target, source) pair: parallel CSR entries add their
    # currents in the network but have no lossless dense or sync representation.
    for src_idx in range(projection.source.n):
        targets = projection.indices[projection.indptr[src_idx] : projection.indptr[src_idx + 1]]
        if len(set(int(t) for t in targets)) != len(targets):
            raise NotImplementedError(
                "Network.to_torch() does not support parallel connections (one source "
                f"{src_idx} reaches one target twice); merge them into one weight"
            )
    if float(getattr(projection, "weight_threshold", 0.0)) != 0.0:
        raise NotImplementedError(
            "Network.to_torch() does not support weight_threshold pruning; use weight_threshold=0.0"
        )


def _uniform(population: Population, attribute: str) -> float:
    """Return a parameter every neuron shares, or refuse a heterogeneous population."""
    values = {float(getattr(neuron, attribute)) for neuron in population.neurons}
    if len(values) != 1:
        raise NotImplementedError(
            f"Network.to_torch() needs one {attribute} per population; "
            f"{population.label!r} has {len(values)}"
        )
    return values.pop()


def _build_population_spec(
    population: Population,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[_PopulationSpec, nn.Module]:
    sample = population.neurons[0]
    voltages = tuple(float(neuron.v) for neuron in population.neurons)
    if population.model_name in ("LapicqueNeuron", "SCLapicqueLIFNeuron"):
        # Every neuron's profile, not the first one's: one shared cell steps them all.
        for neuron in population.neurons:
            if getattr(neuron, "profile", "sc_lif") != "sc_lif":
                raise NotImplementedError(
                    "Network.to_torch() supports LapicqueNeuron only in its SC LIF profile; the "
                    f"{neuron.profile!r} source profile latches one event and has no hard reset"
                )
        cell: nn.Module = _ExactResetCell(
            form="flow",
            decay=math.exp(-_uniform(population, "dt") / _uniform(population, "tau")),
            resistance=_uniform(population, "resistance"),
            dt=_uniform(population, "dt"),
            v_rest=_uniform(population, "v_rest"),
            v_reset=_uniform(population, "v_reset"),
            threshold=_uniform(population, "v_threshold"),
            surrogate_fn=surrogate_fn,
        )
        spec = _PopulationSpec(population, "v", voltages, population.label)
        return spec, cell

    if population.model_name == "StochasticLIFNeuron":
        if float(sample.noise_std) != 0.0:
            raise NotImplementedError(
                "Network.to_torch() supports StochasticLIFNeuron only with noise_std == 0.0"
            )
        if int(sample.refractory_period) != 0:
            raise NotImplementedError(
                "Network.to_torch() supports StochasticLIFNeuron only with refractory_period == 0"
            )
        if sample.entropy_source is not None:
            raise NotImplementedError(
                "Network.to_torch() does not support external entropy_source in StochasticLIFNeuron"
            )
        for neuron in population.neurons:
            if float(neuron.noise_std) != 0.0 or int(neuron.refractory_period) != 0:
                raise NotImplementedError(
                    "Network.to_torch() needs noise_std == 0 and refractory_period == 0 "
                    "for every neuron"
                )
        dt = _uniform(population, "dt")
        # v += -(v - v_rest)·(dt/tau) + (R·I)·dt: the model's forward Euler step, exactly.
        cell = _ExactResetCell(
            form="euler",
            decay=dt / _uniform(population, "tau_mem"),
            resistance=_uniform(population, "resistance"),
            dt=dt,
            v_rest=_uniform(population, "v_rest"),
            v_reset=_uniform(population, "v_reset"),
            threshold=_uniform(population, "v_threshold"),
            surrogate_fn=surrogate_fn,
        )
        spec = _PopulationSpec(population, "v", voltages, population.label)
        return spec, cell

    raise NotImplementedError(
        f"Network.to_torch() does not support model {population.model_name!r} yet"
    )


class NetworkTorchBridge(_TorchModuleBase):
    """Differentiable bridge for a bounded subset of declarative ``Network`` graphs."""

    def __init__(
        self,
        populations: list[Population],
        projections: list[Projection],
        surrogate_fn: Callable[[torch.Tensor], torch.Tensor] = atan_surrogate_custom_op,
    ) -> None:
        super().__init__()
        if not populations:
            raise ValueError("Network.to_torch() requires at least one population")
        for population in populations:
            if population.n <= 0:
                raise ValueError("Network.to_torch() requires populations with n > 0")
        population_ids = {id(population) for population in populations}
        for projection in projections:
            if (
                id(projection.source) not in population_ids
                or id(projection.target) not in population_ids
            ):
                raise ValueError(
                    "Network.to_torch() projection endpoints must belong to the network"
                )

        self.populations = populations
        self.projections = projections
        self.surrogate_fn = surrogate_fn

        self.population_specs: dict[int, _PopulationSpec] = {}
        self.population_cells = nn.ModuleDict()
        for population in populations:
            spec, cell = _build_population_spec(population, surrogate_fn)
            self.population_specs[id(population)] = spec
            self.population_cells[str(id(population))] = cell

        self._input_population_ids = self._find_input_populations()
        self._output_population_ids = self._find_output_populations()
        self.input_dim = sum(
            self.population_specs[pid].population.n for pid in self._input_population_ids
        )
        self.output_dim = sum(
            self.population_specs[pid].population.n for pid in self._output_population_ids
        )
        if self.input_dim <= 0:
            raise ValueError(
                "Network.to_torch() requires at least one input population with no incoming "
                "projections; closed recurrent graphs need an explicit input surface"
            )
        if self.output_dim <= 0:  # pragma: no cover - population width validation guards this.
            raise ValueError("Network.to_torch() resolved an empty output surface")
        output_labels = [self.population_specs[pid].label for pid in self._output_population_ids]
        if len(output_labels) != len(set(output_labels)):
            raise ValueError("Network.to_torch() output population labels must be unique")

        self._projection_names: dict[int, str] = {}
        for index, projection in enumerate(projections):
            if projection.plasticity is not None:
                raise NotImplementedError(
                    "Network.to_torch() does not support plastic projections yet"
                )
            if projection.delay_mode != "none":
                raise NotImplementedError(
                    "Network.to_torch() does not support delayed projections yet"
                )
            _validate_projection_csr(projection)
            name = f"proj_{index}"
            self._projection_names[id(projection)] = name
            self.register_parameter(
                f"{name}_weight",
                nn.Parameter(torch.from_numpy(_csr_to_dense(projection)).clone()),
            )
            self.register_buffer(f"{name}_mask", torch.from_numpy(_csr_mask(projection)))

    def _find_input_populations(self) -> list[int]:
        incoming = {id(pop): 0 for pop in self.populations}
        for projection in self.projections:
            incoming[id(projection.target)] += 1
        return [id(pop) for pop in self.populations if incoming[id(pop)] == 0]

    def _find_output_populations(self) -> list[int]:
        outgoing = {id(pop): 0 for pop in self.populations}
        for projection in self.projections:
            outgoing[id(projection.source)] += 1
        outputs = [id(pop) for pop in self.populations if outgoing[id(pop)] == 0]
        return outputs or [id(pop) for pop in self.populations]

    def _projection_weight(
        self, projection: Projection, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        name = self._projection_names[id(projection)]
        weight = getattr(self, f"{name}_weight")
        mask = getattr(self, f"{name}_mask")
        if not isinstance(weight, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError(
                f"registered projection tensors are corrupted for {name}_weight/{name}_mask"
            )
        masked = weight * mask
        return masked if dtype is None else masked.to(dtype)

    def _initial_state(
        self,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        voltages: dict[int, torch.Tensor] = {}
        spikes: dict[int, torch.Tensor] = {}
        for pid, spec in self.population_specs.items():
            n = spec.population.n
            # Every batch member starts from each neuron's own voltage.
            per_neuron = torch.tensor(spec.init_voltage, dtype=dtype, device=device)
            voltages[pid] = per_neuron.expand(batch, n).clone()
            spikes[pid] = torch.zeros((batch, n), dtype=dtype, device=device)
        return voltages, spikes

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        return_traces: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the differentiable bridge on ``(T, batch, input_dim)`` current input."""
        if inputs.ndim != 3:
            raise ValueError(
                f"Expected inputs with shape (T, batch, input_dim), got {tuple(inputs.shape)}"
            )
        if inputs.shape[2] != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {inputs.shape[2]}")
        if inputs.shape[0] <= 0:
            raise ValueError("Network.to_torch() requires at least one timestep")
        if not torch.is_floating_point(inputs):
            raise ValueError("Network.to_torch() inputs must be floating-point tensors")
        if not torch.isfinite(inputs).all():
            raise ValueError("Network.to_torch() inputs must contain only finite values")

        timesteps, batch, _ = inputs.shape
        device = inputs.device
        dtype = inputs.dtype
        voltages, last_spikes = self._initial_state(batch, device, dtype)
        spike_counts: dict[int, torch.Tensor] = {
            pid: torch.zeros(
                (batch, self.population_specs[pid].population.n), dtype=dtype, device=device
            )
            for pid in self._output_population_ids
        }
        traces: dict[str, list[torch.Tensor]] = {
            self.population_specs[pid].label: [] for pid in self._output_population_ids
        }

        input_offset = 0
        input_slices: dict[int, slice] = {}
        for pid in self._input_population_ids:
            width = self.population_specs[pid].population.n
            input_slices[pid] = slice(input_offset, input_offset + width)
            input_offset += width

        for t in range(timesteps):
            currents: dict[int, torch.Tensor] = {
                pid: torch.zeros_like(v) for pid, v in voltages.items()
            }
            for pid, sl in input_slices.items():
                currents[pid] = currents[pid] + inputs[t, :, sl]

            for projection in self.projections:
                src_pid = id(projection.source)
                tgt_pid = id(projection.target)
                weight = self._projection_weight(projection, dtype)
                currents[tgt_pid] = currents[tgt_pid] + torch.matmul(last_spikes[src_pid], weight.T)

            new_spikes: dict[int, torch.Tensor] = {}
            for population in self.populations:
                pid = id(population)
                cell = self.population_cells[str(pid)]
                spike, new_voltage = cell(currents[pid], voltages[pid])
                voltages[pid] = new_voltage
                new_spikes[pid] = spike
                if pid in spike_counts:
                    spike_counts[pid] = spike_counts[pid] + spike
                    if return_traces:
                        traces[self.population_specs[pid].label].append(spike)

            last_spikes = new_spikes

        counts = torch.cat([spike_counts[pid] for pid in self._output_population_ids], dim=1)
        if not return_traces:
            return counts

        stacked = {
            label: torch.stack(items, dim=0)
            if items
            else torch.empty(0, batch, 0, device=device, dtype=dtype)
            for label, items in traces.items()
        }
        return counts, stacked

    def sync_to_network(self) -> None:
        """Copy learned bridge weights back into the underlying CSR projections."""
        for projection in self.projections:
            dense = self._projection_weight(projection).detach().cpu().numpy()
            for src_idx in range(projection.source.n):
                for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
                    tgt_idx = projection.indices[k]
                    projection.data[k] = float(dense[tgt_idx, src_idx])
