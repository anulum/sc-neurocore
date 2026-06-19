# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Intel Lava / Loihi integration bridge

"""Intel Lava / Loihi integration bridge.

Maps SC-NeuroCore networks to Lava Processes for deployment
on Intel Loihi 2 neuromorphic hardware or Lava CPU simulation.

Requires: pip install lava-nc (optional dependency)

Usage:
    from sc_neurocore.integrations.lava_bridge import (
        SCtoLavaConverter, export_weights_loihi, SCDenseProcess,
    )

The two Lava classes (``SCDenseProcess`` and ``PySCDenseModel``)
are only defined when ``HAS_LAVA`` is True; otherwise they are
absent from the module namespace. Helpers (the converter and the
weight/threshold encoders plus ``LoihiNetworkConfig``) are always
importable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from lava.magma.core.process.process import AbstractProcess
    from lava.magma.core.process.ports.ports import InPort, OutPort
    from lava.magma.core.process.variable import Var
    from lava.magma.core.model.py.model import PyLoihiProcessModel
    from lava.magma.core.model.py.type import LavaPyType
    from lava.magma.core.model.py.ports import PyInPort, PyOutPort
    from lava.magma.core.resources import CPU
    from lava.magma.core.decorator import implements, requires
    from lava.magma.core.sync.protocols.loihi_protocol import LoihiProtocol

    HAS_LAVA = True
except ImportError:
    HAS_LAVA = False


def export_weights_loihi(
    weights: np.ndarray[Any, Any],
    weight_bits: int = 8,
    weight_exp: int = 0,
) -> np.ndarray[Any, Any]:
    """Convert SC probability weights [0,1] to Loihi fixed-point format.

    Loihi uses signed integer weights with configurable precision.
    Maps [0,1] → [-128, 127] for 8-bit weights.
    """
    max_val = (1 << (weight_bits - 1)) - 1
    min_val = -(1 << (weight_bits - 1))
    # SC weights are [0,1], shift to [-1,1] then scale
    scaled = (weights * 2.0 - 1.0) * max_val
    quantised = np.clip(np.round(scaled), min_val, max_val).astype(np.int32)
    encoded: np.ndarray[Any, Any] = quantised * (2**weight_exp)
    return encoded


def loihi_threshold_from_sc(sc_threshold: float, weight_bits: int = 8) -> int:
    """Convert SC normalised threshold to Loihi integer threshold."""
    max_val = (1 << (weight_bits - 1)) - 1
    return int(np.round(sc_threshold * max_val))


@dataclass
class LoihiNetworkConfig:
    """Configuration for a deployed Loihi network."""

    n_inputs: int
    n_outputs: int
    weights: np.ndarray[Any, Any]
    thresholds: np.ndarray[Any, Any]
    weight_bits: int = 8
    weight_exp: int = 0
    decay: int = 128


class SCtoLavaConverter:
    """Convert SC-NeuroCore layer stack to Lava Process network."""

    def __init__(self, weight_bits: int = 8):
        self.weight_bits = weight_bits

    def convert_dense_layer(self, sc_layer: object) -> LoihiNetworkConfig:
        """Convert an SCDenseLayer or VectorizedSCLayer to Loihi config."""
        weights = np.array(sc_layer.weights)  # type: ignore[attr-defined]
        loihi_weights = export_weights_loihi(weights, self.weight_bits)
        thresholds = np.full(weights.shape[0], loihi_threshold_from_sc(1.0, self.weight_bits))
        return LoihiNetworkConfig(
            n_inputs=weights.shape[1],
            n_outputs=weights.shape[0],
            weights=loihi_weights,
            thresholds=thresholds,
            weight_bits=self.weight_bits,
        )

    def convert_training_model(self, spiking_net: object) -> list[LoihiNetworkConfig]:
        """Convert a trained SpikingNet to a list of LoihiNetworkConfigs."""
        configs = []
        sc_weights = spiking_net.to_sc_weights()  # type: ignore[attr-defined]
        for w in sc_weights:
            w_np = w.numpy() if hasattr(w, "numpy") else np.array(w)
            loihi_w = export_weights_loihi(w_np, self.weight_bits)
            n_out, n_in = w_np.shape
            thresholds = np.full(n_out, loihi_threshold_from_sc(1.0, self.weight_bits))
            configs.append(
                LoihiNetworkConfig(
                    n_inputs=n_in,
                    n_outputs=n_out,
                    weights=loihi_w,
                    thresholds=thresholds,
                    weight_bits=self.weight_bits,
                )
            )
        return configs


if HAS_LAVA:

    class SCDenseProcess(AbstractProcess):  # type: ignore[misc]
        """Lava Process wrapping an SC-NeuroCore dense layer."""

        def __init__(self, config: LoihiNetworkConfig):
            super().__init__()
            self.s_in = InPort(shape=(config.n_inputs,))
            self.s_out = OutPort(shape=(config.n_outputs,))
            self.weights = Var(shape=config.weights.shape, init=config.weights)
            self.v = Var(shape=(config.n_outputs,), init=np.zeros(config.n_outputs))
            self.threshold = Var(shape=(config.n_outputs,), init=config.thresholds)
            self.decay = Var(shape=(1,), init=config.decay)

    @implements(proc=SCDenseProcess, protocol=LoihiProtocol)
    @requires(CPU)
    class PySCDenseModel(PyLoihiProcessModel):  # type: ignore[misc]
        s_in: PyInPort = LavaPyType(PyInPort.VEC_DENSE, int)
        s_out: PyOutPort = LavaPyType(PyOutPort.VEC_DENSE, int)
        weights: np.ndarray[Any, Any] = LavaPyType(np.ndarray, int)
        v: np.ndarray[Any, Any] = LavaPyType(np.ndarray, int)
        threshold: np.ndarray[Any, Any] = LavaPyType(np.ndarray, int)
        decay: np.ndarray[Any, Any] = LavaPyType(np.ndarray, int)

        def run_spk(self) -> None:
            spikes_in = self.s_in.recv()
            current = self.weights @ spikes_in
            self.v[:] = (self.v * self.decay[0]) // 256 + current
            spikes_out = (self.v >= self.threshold).astype(int)
            self.v[spikes_out == 1] = 0
            self.s_out.send(spikes_out)
