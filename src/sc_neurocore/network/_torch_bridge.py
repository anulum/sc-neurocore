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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import torch
import torch.nn as nn

from sc_neurocore.training.snn_modules import LapicqueCell
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
    init_voltage: float
    label: str


def _csr_to_dense(projection: Projection) -> np.ndarray[Any, Any]:
    dense = np.zeros((projection.target.n, projection.source.n), dtype=np.float32)
    for src_idx in range(projection.source.n):
        for k in range(projection.indptr[src_idx], projection.indptr[src_idx + 1]):
            tgt_idx = projection.indices[k]
            dense[tgt_idx, src_idx] = float(projection.data[k])
    return dense


def _csr_mask(projection: Projection) -> np.ndarray[Any, Any]:
    mask = np.zeros((projection.target.n, projection.source.n), dtype=np.float32)
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


def _build_population_spec(
    population: Population,
    surrogate_fn: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[_PopulationSpec, nn.Module]:
    sample = population.neurons[0]
    if population.model_name == "LapicqueNeuron":
        cell = LapicqueCell(
            tau=float(sample.tau),
            r=float(sample.resistance),
            dt=float(sample.dt),
            threshold=float(sample.v_threshold),
            v_rest=float(sample.v_rest),
            surrogate_fn=surrogate_fn,
        )
        return (
            _PopulationSpec(
                population=population,
                state_kind="v",
                init_voltage=float(sample.v),
                label=population.label,
            ),
            cell,
        )

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
        if float(sample.v_reset) != float(sample.v_rest):
            raise NotImplementedError(
                "Network.to_torch() supports StochasticLIFNeuron only when v_reset == v_rest"
            )

        # LapicqueCell gain = r * dt / tau, so r must absorb tau_mem to match
        # the discrete StochasticLIF input term resistance * current * dt.
        cell = LapicqueCell(
            tau=float(sample.tau_mem),
            r=float(sample.resistance) * float(sample.tau_mem),
            dt=float(sample.dt),
            threshold=float(sample.v_threshold),
            v_rest=float(sample.v_rest),
            surrogate_fn=surrogate_fn,
        )
        return (
            _PopulationSpec(
                population=population,
                state_kind="v",
                init_voltage=float(sample.v),
                label=population.label,
            ),
            cell,
        )

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
            raise ValueError("Network.to_torch() resolved an empty input surface")
        if self.output_dim <= 0:
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

    def _projection_weight(self, projection: Projection) -> torch.Tensor:
        name = self._projection_names[id(projection)]
        weight = getattr(self, f"{name}_weight")
        mask = getattr(self, f"{name}_mask")
        if not isinstance(weight, torch.Tensor) or not isinstance(mask, torch.Tensor):
            raise TypeError("registered projection tensors are corrupted")
        return weight * mask

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
            voltages[pid] = torch.full(
                (batch, n),
                fill_value=spec.init_voltage,
                dtype=dtype,
                device=device,
            )
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
                weight = self._projection_weight(projection)
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
