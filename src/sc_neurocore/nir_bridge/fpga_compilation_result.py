# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR/ONNX → FPGA network compiler
"""Compilation result contracts for NIR-to-FPGA lowering."""

from dataclasses import dataclass, field

from ..ir.scnir_hdl import SCNIRHDLSourceManifestEntry
from ..ir.scnir_schema import SCNIRDocument


@dataclass
class SCNIRExternalInputManifestEntry:
    """Stable flattened input-bus layout entry for one external source."""

    source: str
    offset: int
    width: int

    def as_dict(self) -> dict[str, int | str]:
        """Return deterministic JSON-ready external input metadata."""
        return {
            "source": self.source,
            "offset": self.offset,
            "width": self.width,
        }


@dataclass(frozen=True)
class FoldedResourceMetrics:
    """Architectural resource summary of a folded (time-multiplexed) interconnect.

    Quantifies what the shared-datapath fold buys versus the direct interconnect's
    one-module-instance-per-neuron unrolling: one processing element per distinct
    neuron type is reused across every neuron of that type (a per-type PE pool), with
    per-neuron state held in BRAM, at the cost of ``cycles_per_tick`` cycles to advance
    the whole network by one timestep.

    Attributes
    ----------
    neurons : int
        Total neurons sharing the datapath across all folded populations.
    state_vars_per_neuron : int
        Widest per-neuron state-variable count across the folded types (the largest
        BRAM word = ``state_vars_per_neuron`` × data width). Equal to the single type's
        count for a homogeneous network.
    pe_instances : int
        Physical processing elements instantiated: one per distinct neuron type. A
        single population, or several populations all of one type, share one PE.
    shared_multipliers : int
        Multipliers in the shared weighted fan-in, summed over external-source columns
        across all populations and reused across each population's neurons. Spiking
        fan-in (recurrent or inter-population) is spike-gated and uses none.
    state_ram_bits : int
        Total BRAM-backed neuron-state storage, in bits, summed over populations
        (each population contributes ``neurons`` × its type's state-var count × data width).
    cycles_per_tick : int
        Clock cycles to advance the whole network by one timestep
        (``neurons`` process cycles + 1 commit cycle).
    direct_neuron_instances : int
        Neuron module instances the direct interconnect would unroll (= ``neurons``);
        the count the fold collapses to ``pe_instances``.
    populations : int
        Number of folded populations sharing the one sequencer and the global spike bus.
    param_rom_bits : int
        Total per-neuron parameter-ROM storage, in bits, for heterogeneous populations
        (each contributes ``neurons`` × its count of per-neuron-varying parameters × data
        width). Zero for a network whose populations all have uniform parameters (the PE
        bakes them). The parameter-space analogue of ``state_ram_bits``.
    """

    neurons: int
    state_vars_per_neuron: int
    pe_instances: int
    shared_multipliers: int
    state_ram_bits: int
    cycles_per_tick: int
    direct_neuron_instances: int
    populations: int = 1
    param_rom_bits: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a deterministic plain-``int`` mapping for manifests/JSON."""
        return {
            "neurons": self.neurons,
            "state_vars_per_neuron": self.state_vars_per_neuron,
            "pe_instances": self.pe_instances,
            "shared_multipliers": self.shared_multipliers,
            "state_ram_bits": self.state_ram_bits,
            "cycles_per_tick": self.cycles_per_tick,
            "direct_neuron_instances": self.direct_neuron_instances,
            "populations": self.populations,
            "param_rom_bits": self.param_rom_bits,
        }


@dataclass
class NetworkCompilationResult:
    """All artefacts from a network-level FPGA compilation.

    Attributes
    ----------
    neuron_modules : dict[str, str]
        Mapping from neuron type to Verilog source.
    weight_rom : str
        Weight ROM Verilog source.
    top_module : str
        Top-level interconnect Verilog source.
    module_name : str
        Top-level module name.
    total_neurons : int
        Total neuron count.
    total_synapses : int
        Total synapse count.
    q_format : str
        Q-format label (e.g. ``"Q8.8"``).
    interconnect : str
        ``"direct"``, ``"aer"``, or ``"folded"`` (the time-multiplexed shared datapath).
    folded_metrics : FoldedResourceMetrics | None
        Architectural fold resource summary when ``interconnect == "folded"``; ``None``
        for the direct/AER paths.
    warnings : list[str]
        Quantisation and compilation warnings.
    scnir_document : SCNIRDocument
        SC-aware metadata document consumed by the compilation artefacts.
    scnir_source_modules : dict[str, str]
        Concrete stochastic source HDL modules keyed by Verilog module name.
    scnir_source_manifest : tuple[SCNIRHDLSourceManifestEntry, ...]
        Deterministic manifest mapping SC-NIR streams to source modules.
    scnir_external_inputs : tuple[SCNIRExternalInputManifestEntry, ...]
        Deterministic flattened input-bus layout for external source names.
    scnir_hierarchy_modules : dict[str, str]
        Standalone SC-NIR hierarchy boundary modules keyed by module name.
    """

    neuron_modules: dict[str, str]
    weight_rom: str
    top_module: str
    module_name: str
    total_neurons: int
    total_synapses: int
    q_format: str
    interconnect: str
    scnir_document: SCNIRDocument
    scnir_source_modules: dict[str, str]
    scnir_source_manifest: tuple[SCNIRHDLSourceManifestEntry, ...]
    scnir_external_inputs: tuple[SCNIRExternalInputManifestEntry, ...]
    scnir_hierarchy_modules: dict[str, str]
    folded_metrics: FoldedResourceMetrics | None = None
    warnings: list[str] = field(default_factory=list)


# Preserve the historical public import and pickle path after extracting these
# contracts from ``fpga_compiler``.
SCNIRExternalInputManifestEntry.__module__ = "sc_neurocore.nir_bridge.fpga_compiler"
FoldedResourceMetrics.__module__ = "sc_neurocore.nir_bridge.fpga_compiler"
NetworkCompilationResult.__module__ = "sc_neurocore.nir_bridge.fpga_compiler"
