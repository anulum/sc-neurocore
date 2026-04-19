# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Edge deployment tooling

"""Edge deployment: power estimation, Sobol sequences, RISC-V config generation,
SC bitstream primitives, LFSR encoder, spiking neurons, network runner,
runtime telemetry, and weight serialization.

Ported from ``tinysc_riscv`` Rust no_std crate.
"""

from sc_neurocore.edge.power_estimator import PowerProfile, Board
from sc_neurocore.edge.sobol import SobolGenerator
from sc_neurocore.edge.deploy import generate_cargo_config, generate_memory_x
from sc_neurocore.edge.bitstream import (
    popcount32,
    popcount_slice,
    sc_and,
    sc_or,
    sc_xor,
    sc_sub,
    sc_mux,
    and_packed,
    mux_packed,
    probability,
    scc,
)
from sc_neurocore.edge.lfsr import Lfsr16
from sc_neurocore.edge.neuron import LifNeuron, IzhikevichNeuron
from sc_neurocore.edge.sc_network import SCLayer, SCNetwork
from sc_neurocore.edge.telemetry import TelemetryRing, LayerTelemetry, DeviceTelemetry
from sc_neurocore.edge.weights import (
    serialize_weights,
    deserialize_weights,
    WeightHeader,
    LayerHeader,
    WEIGHT_MAGIC,
)

__all__ = [
    "PowerProfile",
    "Board",
    "SobolGenerator",
    "generate_cargo_config",
    "generate_memory_x",
    "popcount32",
    "popcount_slice",
    "sc_and",
    "sc_or",
    "sc_xor",
    "sc_sub",
    "sc_mux",
    "and_packed",
    "mux_packed",
    "probability",
    "scc",
    "Lfsr16",
    "LifNeuron",
    "IzhikevichNeuron",
    "SCLayer",
    "SCNetwork",
    "TelemetryRing",
    "LayerTelemetry",
    "DeviceTelemetry",
    "serialize_weights",
    "deserialize_weights",
    "WeightHeader",
    "LayerHeader",
    "WEIGHT_MAGIC",
]
