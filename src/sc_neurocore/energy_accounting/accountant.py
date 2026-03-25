# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-spike energy accounting system

"""Per-spike, per-synapse, per-timestep energy accounting mapped to hardware.

"This classification cost 4.2 nJ: 3.1 nJ layer 2 synaptic ops,
0.8 nJ layer 1 membrane updates, 0.3 nJ memory access."

A 2025 paper identifies a gap: no energy metric is both Accessible
and High Fidelity. This module fills that gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardwareCostModel:
    """Energy costs for a specific hardware target.

    All values in picojoules (pJ).
    """

    name: str
    # Per-operation costs (pJ)
    synop_pj: float = 23.6  # synaptic operation
    membrane_update_pj: float = 1.0  # LIF membrane update
    spike_generation_pj: float = 0.5  # comparator + reset
    memory_read_pj: float = 5.0  # weight read from SRAM
    memory_write_pj: float = 8.0  # weight write (for plasticity)
    routing_pj: float = 2.0  # inter-core spike routing
    # Static
    leakage_pw_per_neuron: float = 10.0  # picoWatts per neuron


HARDWARE_COSTS: dict[str, HardwareCostModel] = {
    "loihi2": HardwareCostModel(
        name="loihi2",
        synop_pj=23.6,
        membrane_update_pj=1.2,
        spike_generation_pj=0.5,
        memory_read_pj=4.0,
        routing_pj=3.0,
    ),
    "akida": HardwareCostModel(
        name="akida",
        synop_pj=10.0,
        membrane_update_pj=0.8,
        spike_generation_pj=0.3,
        memory_read_pj=3.0,
        routing_pj=1.0,
    ),
    "fpga_ice40": HardwareCostModel(
        name="fpga_ice40",
        synop_pj=50.0,
        membrane_update_pj=5.0,
        spike_generation_pj=2.0,
        memory_read_pj=10.0,
        routing_pj=0.5,
    ),
    "fpga_artix7": HardwareCostModel(
        name="fpga_artix7",
        synop_pj=30.0,
        membrane_update_pj=3.0,
        spike_generation_pj=1.0,
        memory_read_pj=8.0,
        routing_pj=0.3,
    ),
    "analog_28nm": HardwareCostModel(
        name="analog_28nm",
        synop_pj=1.6,
        membrane_update_pj=0.2,
        spike_generation_pj=0.1,
        memory_read_pj=0.5,
        routing_pj=0.3,
    ),
}


@dataclass
class LayerEnergy:
    """Energy breakdown for one layer."""

    name: str
    synop_energy_pj: float = 0.0
    membrane_energy_pj: float = 0.0
    spike_gen_energy_pj: float = 0.0
    memory_energy_pj: float = 0.0
    total_pj: float = 0.0
    n_synops: int = 0
    n_spikes: int = 0
    n_membrane_updates: int = 0


@dataclass
class EnergyReport:
    """Complete energy accounting report."""

    hardware: str
    layers: list[LayerEnergy] = field(default_factory=list)
    total_energy_pj: float = 0.0
    total_energy_nj: float = 0.0
    routing_energy_pj: float = 0.0

    def summary(self) -> str:
        lines = [
            f"Energy Report [{self.hardware}]: {self.total_energy_nj:.2f} nJ total",
            "",
        ]
        for le in self.layers:
            pct = le.total_pj / max(self.total_energy_pj, 1e-12) * 100
            lines.append(
                f"  {le.name}: {le.total_pj:.1f} pJ ({pct:.0f}%) — "
                f"{le.n_synops} synops, {le.n_spikes} spikes"
            )
        lines.append(f"  Routing: {self.routing_energy_pj:.1f} pJ")
        return "\n".join(lines)

    @property
    def dominant_layer(self) -> str | None:
        if not self.layers:
            return None
        return max(self.layers, key=lambda l: l.total_pj).name  # pragma: no cover

    @property
    def energy_per_spike_pj(self) -> float:
        total_spikes = sum(l.n_spikes for l in self.layers)
        if total_spikes == 0:
            return 0.0
        return self.total_energy_pj / total_spikes


class EnergyAccountant:
    """Per-spike energy accounting system.

    Parameters
    ----------
    hardware : str or HardwareCostModel
        Target hardware for cost mapping.
    """

    def __init__(self, hardware: str | HardwareCostModel = "loihi2"):
        if isinstance(hardware, str):
            self.cost_model = HARDWARE_COSTS.get(hardware)
            if self.cost_model is None:
                raise ValueError(
                    f"Unknown hardware '{hardware}'. Available: {list(HARDWARE_COSTS.keys())}"
                )
        else:
            self.cost_model = hardware

    def account(
        self,
        layer_names: list[str],
        layer_sizes: list[tuple[int, int]],
        spike_counts: list[int],
        n_timesteps: int,
    ) -> EnergyReport:
        """Compute energy breakdown from spike activity.

        Parameters
        ----------
        layer_names : list of str
        layer_sizes : list of (n_inputs, n_neurons)
        spike_counts : list of int
            Total spikes per layer across all timesteps.
        n_timesteps : int

        Returns
        -------
        EnergyReport
        """
        c = self.cost_model
        report = EnergyReport(hardware=c.name)
        total_spikes_all = 0

        for name, (n_in, n_out), n_spikes in zip(layer_names, layer_sizes, spike_counts):
            # Synaptic operations: each spike activates n_out synapses
            n_synops = n_spikes * n_in
            synop_e = n_synops * c.synop_pj

            # Membrane updates: all neurons updated every timestep
            n_mem = n_out * n_timesteps
            mem_e = n_mem * c.membrane_update_pj

            # Spike generation
            spike_e = n_spikes * c.spike_generation_pj

            # Memory: each synop reads a weight
            mem_read_e = n_synops * c.memory_read_pj

            total = synop_e + mem_e + spike_e + mem_read_e

            report.layers.append(
                LayerEnergy(
                    name=name,
                    synop_energy_pj=synop_e,
                    membrane_energy_pj=mem_e,
                    spike_gen_energy_pj=spike_e,
                    memory_energy_pj=mem_read_e,
                    total_pj=total,
                    n_synops=n_synops,
                    n_spikes=n_spikes,
                    n_membrane_updates=n_mem,
                )
            )
            total_spikes_all += n_spikes

        # Routing energy: each spike routed between layers
        report.routing_energy_pj = total_spikes_all * c.routing_pj

        report.total_energy_pj = sum(l.total_pj for l in report.layers) + report.routing_energy_pj
        report.total_energy_nj = report.total_energy_pj / 1000.0

        return report
