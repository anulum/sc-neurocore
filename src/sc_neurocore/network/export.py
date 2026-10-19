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

The format is signed fixed point, ``Q8.8`` by default (``data_width=16``,
``fraction=8``). Every lowered parameter and the initial voltage are checked
at that format by :func:`~sc_neurocore.compiler.hardware_numeric_contract.hardware_numeric_contract`
before anything is written: a value outside the format's range (it would wrap
to another value) or a non-zero value that rounds to zero refuses the export
and names the quantity; a wider format can be chosen instead. Values within
range are rounded to the format, and a run can still saturate.

Projections are not exported: the top exposes one current word per neuron, in
the chosen format, and one spike bit per neuron.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

from sc_neurocore.exceptions import SCHardwareError

if TYPE_CHECKING:
    from sc_neurocore.neurons.equation_builder import EquationNeuron

    from .network import Network
    from .population import Population

DATA_WIDTH = 16
"""Default word width in bits (``Q8.8`` with :data:`FRACTION`)."""
FRACTION = 8
"""Default number of fractional bits."""

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
    common = {
        "v_rest": _uniform(pop, "v_rest"),
        "v_reset": _uniform(pop, "v_reset"),
        "v_threshold": _uniform(pop, "v_threshold"),
    }
    if name in ("SCLapicqueLIFNeuron", "LapicqueNeuron"):
        # Every neuron's profile, not the first one's: one module steps them all.
        for neuron in pop.neurons:
            if getattr(neuron, "profile", "sc_lif") != "sc_lif":
                raise SCHardwareError(
                    f"population {pop.label!r}: the {neuron.profile!r} Lapicque profile "
                    "latches one event and has no reset; only the SC LIF profile is exported"
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
    for quantity, value in parameters.items():
        if not math.isfinite(value):
            raise SCHardwareError(
                f"population {pop.label!r}: the lowered parameter {quantity}={value!r} is not "
                "finite; no fixed-point format holds it"
            )
    return EquationNeuron(
        equations=equations,
        parameters=parameters,
        state={"v": _uniform(pop, "v")},
        dt=1.0,
        method="map",
        threshold="v >= v_threshold",
        reset={"v": "v_reset"},
    )


def _check_format(data_width: int, fraction: int) -> None:
    """Refuse a word width or fraction the exporter cannot lower to."""
    for name, value in (("data_width", data_width), ("fraction", fraction)):
        if type(value) is not int:
            raise SCHardwareError(f"{name} must be an integer, got {value!r}")
    if not 0 < fraction < data_width:
        raise SCHardwareError(
            f"need 0 < fraction < data_width, got data_width={data_width}, fraction={fraction}"
        )


def _check_exportable(network: Network, data_width: int, fraction: int) -> list[EquationNeuron]:
    """Return one equation neuron per population, refusing anything not exported.

    Each neuron's lowered parameters and initial voltage must be representable
    at the format: none outside its range and no non-zero value rounded to zero.
    """
    from sc_neurocore.compiler.hardware_numeric_contract import hardware_numeric_contract
    from sc_neurocore.compiler.q_format import QFormat

    _check_format(data_width, fraction)
    if not network.populations:
        raise SCHardwareError("an empty network has no populations to export")
    neurons = [_population_neuron(pop) for pop in network.populations]
    q_format = QFormat(data_width - fraction, fraction)
    for pop, neuron in zip(network.populations, neurons, strict=True):
        contract = hardware_numeric_contract(neuron, q_format)
        if not contract.representable:
            raise SCHardwareError(f"population {pop.label!r}: {contract.refusal()}")
    return neurons


def _emit_top(network: Network, target: str, data_width: int, fraction: int) -> str:
    """Emit the neuron modules and the top that instantiates one per neuron."""
    from sc_neurocore.compiler.verilog_compiler import compile_to_verilog

    neurons = _check_exportable(network, data_width, fraction)
    lines = [f"// Auto-generated by SC-NeuroCore — target: {target}", ""]
    for i, neuron in enumerate(neurons):
        lines.append(compile_to_verilog(neuron, f"sc_net_pop{i}_neuron", data_width, fraction))
        lines.append("")
    ports = ["    input wire clk", "    input wire rst_n"]
    for i, pop in enumerate(network.populations):
        ports.append(f"    input wire [{pop.n * data_width - 1}:0] current_{i}")
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
                    f"        .I_t(current_{i}[{j * data_width} +: {data_width}]),",
                    f"        .spike_out(spike_{i}[{j}]),",
                    "        .v_out()",
                    "    );",
                ]
            )
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def export_verilog(
    network: Network,
    output_dir: str,
    target: str = "ice40",
    *,
    data_width: int = DATA_WIDTH,
    fraction: int = FRACTION,
) -> str:
    """Export the network's populations to Verilog files.

    Parameters
    ----------
    network : Network
        A network whose populations use supported models (see the module docs).
    output_dir : str
        Directory for ``sc_network_top.v`` and ``params.vh``; created if needed.
    target : str, default "ice40"
        Label written into the files.
    data_width : int, default 16
        Signed word width in bits of every parameter, state and current word.
    fraction : int, default 8
        Fractional bits of that word (``Q8.8`` by default).

    Returns
    -------
    str
        Path of ``sc_network_top.v``.

    Raises
    ------
    SCHardwareError
        For an empty network, an empty population, an unsupported model or
        configuration, parameters that differ between neurons of a
        population, an invalid format, or a lowered parameter or initial
        voltage the format cannot represent; nothing is written.
    """
    top_v = _emit_top(network, target, data_width, fraction)  # every refusal, before any write
    os.makedirs(output_dir, exist_ok=True)
    top_path = os.path.join(output_dir, "sc_network_top.v")
    with open(top_path, "w") as f:
        f.write(top_v)
    params_path = os.path.join(output_dir, "params.vh")
    with open(params_path, "w") as f:
        f.write(f"// SC-NeuroCore network parameters — target: {target}\n")
        f.write(f"`define SC_NET_DATA_WIDTH {data_width}\n`define SC_NET_FRACTION {fraction}\n")
        for i, pop in enumerate(network.populations):
            f.write(f"`define POP_{i}_SIZE {pop.n}\n")
    return top_path
