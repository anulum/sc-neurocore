# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Evolutionary NIR, Verilog, and photonic emission

"""Emit evolutionary genomes into deployment artefact representations."""

from __future__ import annotations

import textwrap
from typing import Any, Dict, Optional

import numpy as np

from sc_neurocore.evo_substrate.genome import Genome


class OrganismEmitter:
    """Emits evolved organisms as NIR graph or Verilog."""

    @staticmethod
    def to_nir(genome: Genome) -> Dict[str, Any]:
        """Emit a simplified NIR-compatible graph dict."""
        nodes = {}
        for i in range(genome.topology.num_neurons):
            nodes[f"n{i}"] = {
                "type": "ArcaneNeuron",
                "tau_fast": genome.neuron.tau_fast,
                "tau_work": genome.neuron.tau_work,
                "tau_deep": genome.neuron.tau_deep,
                "theta": genome.neuron.theta,
                "gamma": genome.neuron.gamma,
                "delta_conf": genome.neuron.delta_conf,
                "kappa": genome.neuron.kappa,
                "w_inh": genome.neuron.w_inh,
            }
        edges = []
        rng = np.random.default_rng(genome.weight_seed)
        for i in range(genome.topology.num_neurons):
            for j in range(genome.topology.num_neurons):
                if i != j and rng.random() < genome.topology.connectivity:
                    edges.append(
                        {"from": f"n{i}", "to": f"n{j}", "weight_q88": int(rng.integers(0, 256))}
                    )
        return {
            "genome_id": genome.genome_id,
            "generation": genome.generation,
            "nodes": nodes,
            "edges": edges,
            "bitstream_length": genome.topology.bitstream_length,
        }

    @staticmethod
    def to_verilog(genome: Genome, module_name: Optional[str] = None) -> str:
        """Emit Verilog wrapper for the organism."""
        name = module_name or f"sc_organism_{genome.genome_id[:8]}"
        n = genome.topology.num_neurons
        bs = genome.topology.bitstream_length
        return textwrap.dedent(f"""\
// SC-NeuroCore — Evolved Organism: {genome.genome_id}
// Generation: {genome.generation} | Neurons: {n} | Bitstream: {bs}

module {name} #(
    parameter NUM_NEURONS = {n},
    parameter BITSTREAM_W = {bs},
    parameter TAU_FAST    = {int(genome.neuron.tau_fast)},
    parameter TAU_WORK    = {int(genome.neuron.tau_work)},
    parameter THETA_Q88   = {int(genome.neuron.theta * 256)}
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire [BITSTREAM_W-1:0]  sc_input  [0:NUM_NEURONS-1],
    output wire [BITSTREAM_W-1:0]  sc_output [0:NUM_NEURONS-1],
    output wire [NUM_NEURONS-1:0]  spike_out
);

    genvar i;
    generate
        for (i = 0; i < NUM_NEURONS; i = i + 1) begin : neuron_gen
            sc_lif_neuron #(
                .BITSTREAM_W(BITSTREAM_W),
                .THRESHOLD(THETA_Q88)
            ) u_neuron (
                .clk(clk),
                .rst_n(rst_n),
                .bitstream_in(sc_input[i]),
                .bitstream_out(sc_output[i]),
                .spike(spike_out[i])
            );
        end
    endgenerate

endmodule
""")

    @staticmethod
    def to_photonic_netlist(genome: Genome, pml_layers: int = 12) -> Dict[str, Any]:
        """Emit a photonic netlist compatible with the optics PhotonicCompiler."""
        return {
            "version": "1.0",
            "metadata": {
                "genome_id": genome.genome_id,
                "generation": genome.generation,
                "num_neurons": genome.topology.num_neurons,
            },
            "parameters": {
                "wavelength": 1.55e-6,
                "n_core": 3.48,
                "n_clad": 1.44,
                "pml_layers": pml_layers,
            },
            "waveguides": [
                {"id": f"wg_{i}", "width": 0.5, "length": 10.0}
                for i in range(genome.topology.num_neurons)
            ],
        }


__all__ = ["OrganismEmitter"]
