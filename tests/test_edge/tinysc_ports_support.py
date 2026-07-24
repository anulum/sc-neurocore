# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Support for former test_tinysc_ports.py

from __future__ import annotations

"""Comprehensive tests mirroring the Rust test suites from tinysc_riscv."""
import pytest
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
    MASK32,
)
from sc_neurocore.edge.lfsr import Lfsr16
from sc_neurocore.edge.neuron import LifNeuron, IzhikevichNeuron
from sc_neurocore.edge.sc_network import SCLayer, SCNetwork
from sc_neurocore.edge.telemetry import TelemetryRing, DeviceTelemetry
from sc_neurocore.edge.weights import (
    serialize_weights,
    deserialize_weights,
    WeightHeader,
    WEIGHT_MAGIC,
)

__all__ = [
    "pytest",
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
    "MASK32",
    "Lfsr16",
    "LifNeuron",
    "IzhikevichNeuron",
    "SCLayer",
    "SCNetwork",
    "TelemetryRing",
    "DeviceTelemetry",
    "serialize_weights",
    "deserialize_weights",
    "WeightHeader",
    "WEIGHT_MAGIC",
]
