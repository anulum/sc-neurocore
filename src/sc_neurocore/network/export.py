# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Verilog export for LIF-based networks

"""Verilog export of a network's neuron populations, one compiled neuron per instance.

Each population of a supported model is lowered to a neuron module by the
equation compiler (:func:`~sc_neurocore.compiler.verilog_compiler.compile_to_verilog`)
from an :class:`~sc_neurocore.neurons.equation_builder.EquationNeuron` that
steps exactly as the model does, and the top instantiates one module per
neuron. Supported models: the SC LIF Lapicque profile (``SCLapicqueLIFNeuron``,
or ``LapicqueNeuron`` in that profile; exact flow ``v_inf + (v - v_inf)·e^(-dt/tau)``)
and ``StochasticLIFNeuron`` without noise or refractory period (forward Euler),
both with the models' ``>=`` threshold and hard reset. Every neuron of a
population must share its parameters and initial voltage. Anything else, an
empty network or an empty population is refused before a file is written.
Projections are not exported: the top exposes one ``Q8.8`` current word per
neuron and one spike bit per neuron.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any

from sc_neurocore.exceptions import SCHardwareError

if TYPE_CHECKING:
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    from .network import Network
    from .population import Population

DATA_WIDTH = 16
FRACTION = 8

_SUPPORTED = ("SCLapicqueLIFNeuron", "LapicqueNeuron", "StochasticLIFNeuron")


def _uniform(pop: Population, attribute: str) -> float:
    """Return a value every neuron of the population shares, or refuse."""
    values = {float(getattr(neuron, attribute)) for neuron in pop.neurons}
    if len(values) != 1:
        raise SCHardwareError(
            f"population {pop.label!r}: export needs one {attribute} for every neuron, "
            f"got {len(values)} values"
        )
    return values.pop()


def _population_neuron(pop: Population) -> EquationNeuron:
    """Return the equation neuron that steps as the population's model, or refuse."""
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    name = pop._model_cls.__name__
    if name not in _SUPPORTED:
        raise SCHardwareError(
            f"Model '{name}' cannot be exported to Verilog. Supported: {sorted(_SUPPORTED)}"
        )
    if pop.n <= 0:
        raise SCHardwareError(f"population {pop.label!r} has no neurons to export")
    sample: Any = pop.neurons[0]
    common = {
        "v_rest": _uniform(pop, "v_rest"),
        "v_reset": _uniform(pop, "v_reset"),
        "v_threshold": _uniform(pop, "v_threshold"),
    }
    if name in ("SCLapicqueLIFNeuron", "LapicqueNeuron"):
        if getattr(sample, "profile", "sc_lif") != "sc_lif":
            raise SCHardwareError(
                f"population {pop.label!r}: the {sample.profile!r} Lapicque profile latches "
                "one event and has no reset; only the SC LIF profile is exported"
            )
        decay = math.exp(-_uniform(pop, "dt") / _uniform(pop, "tau"))
        parameters = {**common, "decay": decay, "r": _uniform(pop, "resistance")}
        update = "v_inf + (v - v_inf) * decay"
        equations = {"v": update.replace("v_inf", "(v_rest + r * I)")}
    else:
        for neuron in pop.neurons:
            if float(neuron.noise_std) != 0.0 or int(neuron.refractory_period) != 0:
                raise SCHardwareError(
                    f"population {pop.label!r}: StochasticLIFNeuron is exported only with "
                    "noise_std == 0 and refractory_period == 0"
                )
            if neuron.entropy_source is not None:
                raise SCHardwareError(
                    f"population {pop.label!r}: an external entropy_source is not exported"
                )
        dt = _uniform(pop, "dt")
        parameters = {
            **common,
            "leak": dt / _uniform(pop, "tau_mem"),
            "gain": _uniform(pop, "resistance") * dt,
        }
        equations = {"v": "v - (v - v_rest) * leak + gain * I"}
    return EquationNeuron(
        equations=equations,
        parameters=parameters,
        state={"v": _uniform(pop, "v")},
        dt=1.0,
        method="map",
        threshold="v >= v_threshold",
        reset={"v": "v_reset"},
    )


def _check_exportable(network: Network) -> list[EquationNeuron]:
    """Return one equation neuron per population, refusing anything not exported."""
    if not network.populations:
        raise SCHardwareError("an empty network has no populations to export")
    return [_population_neuron(pop) for pop in network.populations]


def _emit_top(network: Network, target: str) -> str:
    """Emit the neuron modules and the top that instantiates one per neuron."""
    from sc_neurocore.compiler.verilog_compiler import compile_to_verilog

    neurons = _check_exportable(network)
    lines = [f"// Auto-generated by SC-NeuroCore — target: {target}", ""]
    for i, neuron in enumerate(neurons):
        lines.append(compile_to_verilog(neuron, f"sc_net_pop{i}_neuron", DATA_WIDTH, FRACTION))
        lines.append("")
    ports = ["    input wire clk", "    input wire rst_n"]
    for i, pop in enumerate(network.populations):
        ports.append(f"    input wire [{pop.n * DATA_WIDTH - 1}:0] current_{i}")
    for i, pop in enumerate(network.populations):
        ports.append(f"    output wire [{pop.n - 1}:0] spike_{i}")
    lines.append("module sc_network_top (")
    lines.append(",\n".join(ports))
    lines.append(");")
    for i, pop in enumerate(network.populations):
        lines.append(f"    // Population {i}: {pop.label} ({pop.n} neurons)")
        for j in range(pop.n):
            lines.extend(
                [
                    f"    sc_net_pop{i}_neuron pop_{i}_n{j} (",
                    "        .clk(clk),",
                    "        .rst_n(rst_n),",
                    f"        .I_t(current_{i}[{j * DATA_WIDTH} +: {DATA_WIDTH}]),",
                    f"        .spike_out(spike_{i}[{j}]),",
                    "        .v_out()",
                    "    );",
                ]
            )
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def export_verilog(network: Network, output_dir: str, target: str = "ice40") -> str:
    """Export the network's populations to Verilog files.

    Parameters
    ----------
    network : Network
        A network whose populations use supported models (see the module docs).
    output_dir : str
        Directory for ``sc_network_top.v`` and ``params.vh``; created if needed.
    target : str, default "ice40"
        Label written into the files.

    Returns
    -------
    str
        Path of ``sc_network_top.v``.

    Raises
    ------
    SCHardwareError
        For an empty network, an empty population, an unsupported model or
        configuration, or parameters that differ between neurons of a
        population; nothing is written.
    """
    top_v = _emit_top(network, target)  # every refusal happens here, before any write
    os.makedirs(output_dir, exist_ok=True)
    top_path = os.path.join(output_dir, "sc_network_top.v")
    with open(top_path, "w") as f:
        f.write(top_v)
    params_path = os.path.join(output_dir, "params.vh")
    with open(params_path, "w") as f:
        f.write(f"// SC-NeuroCore network parameters — target: {target}\n")
        f.write(f"`define SC_NET_DATA_WIDTH {DATA_WIDTH}\n`define SC_NET_FRACTION {FRACTION}\n")
        for i, pop in enumerate(network.populations):
            f.write(f"`define POP_{i}_SIZE {pop.n}\n")
    return top_path
