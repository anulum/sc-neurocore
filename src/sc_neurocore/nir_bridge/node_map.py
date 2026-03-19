# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR node type → SC-NeuroCore primitive factories

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e

from ..neurons.stochastic_lif import StochasticLIFNeuron


@dataclass
class SCInputNode:
    """Graph entry point — passes input through unchanged."""

    name: str
    shape: tuple

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x


@dataclass
class SCOutputNode:
    """Graph exit point — collects output."""

    name: str
    shape: tuple
    last_output: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.last_output = x
        return x


@dataclass
class SCLIFNode:
    """LIF neuron mapped from NIR LIF primitive.

    NIR LIF: tau*dv/dt = (v_leak - v) + R*I, spike when v > v_threshold
    Maps directly to StochasticLIFNeuron.
    """

    name: str
    neurons: list[StochasticLIFNeuron]

    @classmethod
    def from_nir(cls, name: str, node: nir.LIF) -> SCLIFNode:
        tau = np.atleast_1d(node.tau).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = np.atleast_1d(node.v_reset).flatten() if node.v_reset is not None else v_leak

        n = len(tau)
        neurons = []
        for i in range(n):
            neurons.append(
                StochasticLIFNeuron(
                    tau_mem=float(tau[i]),
                    resistance=float(r[i]),
                    v_rest=float(v_leak[i]),
                    v_threshold=float(v_threshold[i]),
                    v_reset=float(v_reset[i]),
                    noise_std=0.0,
                )
            )
        return cls(name=name, neurons=neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()
        spikes = np.zeros(len(self.neurons), dtype=np.float64)
        for i, neuron in enumerate(self.neurons):
            current = float(x[i]) if i < len(x) else 0.0
            spikes[i] = float(neuron.step(current))
        return spikes

    def reset(self):
        for n in self.neurons:
            n.reset_state()


@dataclass
class SCIFNode:
    """IF neuron — integrator with threshold, no leak.

    NIR IF: dv/dt = R*I, spike when v > v_threshold
    """

    name: str
    n_neurons: int
    r: np.ndarray
    v_threshold: np.ndarray
    v_reset: np.ndarray
    v: np.ndarray | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.IF) -> SCIFNode:
        r = np.atleast_1d(node.r).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = (
            np.atleast_1d(node.v_reset).flatten() if node.v_reset is not None else np.zeros_like(r)
        )
        return cls(name=name, n_neurons=len(r), r=r, v_threshold=v_threshold, v_reset=v_reset)

    def __post_init__(self):
        if self.v is None:
            self.v = np.zeros(self.n_neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()[: self.n_neurons]
        self.v += self.r * x
        spikes = (self.v >= self.v_threshold).astype(np.float64)
        self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self):
        self.v = np.zeros(self.n_neurons)


@dataclass
class SCLINode:
    """Leaky integrator — LIF without threshold.

    NIR LI: tau*dv/dt = (v_leak - v) + R*I
    """

    name: str
    n_neurons: int
    tau: np.ndarray
    r: np.ndarray
    v_leak: np.ndarray
    v: np.ndarray | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.LI, dt: float = 1.0) -> SCLINode:
        tau = np.atleast_1d(node.tau).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        return cls(name=name, n_neurons=len(tau), tau=tau, r=r, v_leak=v_leak, dt=dt)

    def __post_init__(self):
        if self.v is None:
            self.v = self.v_leak.copy()

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()[: self.n_neurons]
        dv = (self.v_leak - self.v + self.r * x) * (self.dt / self.tau)
        self.v += dv
        return self.v.copy()

    def reset(self):
        self.v = self.v_leak.copy()


@dataclass
class SCAffineNode:
    """Dense linear transform with bias: y = Wx + b

    Uses VectorizedSCLayer for SC-native computation when in SC mode,
    or plain matrix multiply for float mode.
    """

    name: str
    weight: np.ndarray
    bias: np.ndarray

    @classmethod
    def from_nir(cls, name: str, node: nir.Affine) -> SCAffineNode:
        return cls(name=name, weight=node.weight, bias=node.bias)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()
        return self.weight @ x + self.bias


@dataclass
class SCLinearNode:
    """Matrix multiply without bias: y = Wx"""

    name: str
    weight: np.ndarray

    @classmethod
    def from_nir(cls, name: str, node: nir.Linear) -> SCLinearNode:
        return cls(name=name, weight=node.weight)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()
        return self.weight @ x


@dataclass
class SCScaleNode:
    """Element-wise scaling: y = s * x"""

    name: str
    scale: np.ndarray

    @classmethod
    def from_nir(cls, name: str, node: nir.Scale) -> SCScaleNode:
        return cls(name=name, scale=node.scale)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.scale * x


@dataclass
class SCThresholdNode:
    """Spike threshold: y = 1 if x >= threshold else 0"""

    name: str
    threshold: np.ndarray

    @classmethod
    def from_nir(cls, name: str, node: nir.Threshold) -> SCThresholdNode:
        return cls(name=name, threshold=node.threshold)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return (x >= self.threshold).astype(np.float64)


@dataclass
class SCFlattenNode:
    """Reshape tensor — flatten dimensions."""

    name: str
    start_dim: int
    end_dim: int

    @classmethod
    def from_nir(cls, name: str, node: nir.Flatten) -> SCFlattenNode:
        return cls(name=name, start_dim=node.start_dim, end_dim=node.end_dim)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()


@dataclass
class SCIntegratorNode:
    """Pure integrator: dv/dt = R*I (no leak, no threshold)"""

    name: str
    r: np.ndarray
    v: np.ndarray | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.I) -> SCIntegratorNode:
        r = np.atleast_1d(node.r).flatten()
        return cls(name=name, r=r)

    def __post_init__(self):
        if self.v is None:
            self.v = np.zeros_like(self.r)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()[: len(self.r)]
        self.v += self.r * x
        return self.v.copy()

    def reset(self):
        self.v = np.zeros_like(self.r)


# NIR type → factory function
NODE_MAP: dict[type, Any] = {
    nir.Input: lambda name, node, **kw: SCInputNode(
        name=name,
        shape=tuple(int(x) for x in next(iter(node.input_type.values())).flatten())
        if node.input_type
        else (),
    ),
    nir.Output: lambda name, node, **kw: SCOutputNode(
        name=name,
        shape=tuple(int(x) for x in next(iter(node.output_type.values())).flatten())
        if node.output_type
        else (),
    ),
    nir.LIF: lambda name, node, **kw: SCLIFNode.from_nir(name, node),
    nir.IF: lambda name, node, **kw: SCIFNode.from_nir(name, node),
    nir.LI: lambda name, node, **kw: SCLINode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.I: lambda name, node, **kw: SCIntegratorNode.from_nir(name, node),
    nir.Affine: lambda name, node, **kw: SCAffineNode.from_nir(name, node),
    nir.Linear: lambda name, node, **kw: SCLinearNode.from_nir(name, node),
    nir.Scale: lambda name, node, **kw: SCScaleNode.from_nir(name, node),
    nir.Threshold: lambda name, node, **kw: SCThresholdNode.from_nir(name, node),
    nir.Flatten: lambda name, node, **kw: SCFlattenNode.from_nir(name, node),
}


def map_node(name: str, node: nir.NIRNode, **kwargs) -> Any:
    """Convert a single NIR node to its SC-NeuroCore equivalent."""
    factory = NODE_MAP.get(type(node))
    if factory is None:
        raise NotImplementedError(
            f"NIR node type {type(node).__name__} not yet supported (node: {name!r})"
        )
    return factory(name, node, **kwargs)
