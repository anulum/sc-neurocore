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
    """Dense linear transform with bias: y = Wx + b"""

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
        x = np.asarray(x)
        if x.ndim == 0:
            if self.start_dim not in (0, -1) or self.end_dim not in (0, -1):
                raise ValueError(
                    f"Invalid flatten dims {self.start_dim}:{self.end_dim} for shape {x.shape}"
                )
            return x.reshape(1)

        start = self.start_dim if self.start_dim >= 0 else x.ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else x.ndim + self.end_dim
        if not 0 <= start < x.ndim or not 0 <= end < x.ndim or start > end:
            raise ValueError(
                f"Invalid flatten dims {self.start_dim}:{self.end_dim} for shape {x.shape}"
            )
        if start == end:
            return x.copy()

        merged = int(np.prod(x.shape[start : end + 1], dtype=np.int64))
        new_shape = x.shape[:start] + (merged,) + x.shape[end + 1 :]
        return x.reshape(new_shape)


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


@dataclass
class SCDelayNode:
    """Temporal delay: output = input(t - delay).

    NIR Delay: I(t - tau). Implemented as a circular buffer per element.
    Delay values are rounded to integer timesteps.
    """

    name: str
    delay_steps: np.ndarray
    _buffers: list[list[np.ndarray]] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.Delay, dt: float = 1.0) -> SCDelayNode:
        delay = np.atleast_1d(node.delay).flatten()
        steps = np.maximum(np.round(delay / dt).astype(int), 1)
        return cls(name=name, delay_steps=steps)

    def __post_init__(self):
        if self._buffers is None:
            self._buffers = [[np.zeros(1) for _ in range(int(d))] for d in self.delay_steps]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()
        out = np.zeros(len(self.delay_steps))
        for i, buf in enumerate(self._buffers):
            out[i] = buf[0][0]
            buf.append(np.array([float(x[i]) if i < len(x) else 0.0]))
            buf.pop(0)
        return out

    def reset(self):
        self._buffers = [[np.zeros(1) for _ in range(int(d))] for d in self.delay_steps]


@dataclass
class SCCubaLIFNode:
    """Current-based LIF with synaptic filter.

    NIR CubaLIF: tau_syn * dI_syn/dt = -I_syn + w_in * I
                 tau_mem * dv/dt = (v_leak - v) + R * I_syn
                 spike when v > v_threshold, reset to v_reset
    """

    name: str
    n_neurons: int
    tau_syn: np.ndarray
    tau_mem: np.ndarray
    r: np.ndarray
    v_leak: np.ndarray
    v_threshold: np.ndarray
    v_reset: np.ndarray
    w_in: np.ndarray
    v: np.ndarray | None = None
    i_syn: np.ndarray | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.CubaLIF, dt: float = 1.0) -> SCCubaLIFNode:
        tau_syn = np.atleast_1d(node.tau_syn).flatten()
        tau_mem = np.atleast_1d(node.tau_mem).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = (
            np.atleast_1d(node.v_reset).flatten() if node.v_reset is not None else v_leak.copy()
        )
        w_in = np.atleast_1d(node.w_in).flatten()
        return cls(
            name=name,
            n_neurons=len(tau_mem),
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            r=r,
            v_leak=v_leak,
            v_threshold=v_threshold,
            v_reset=v_reset,
            w_in=w_in,
            dt=dt,
        )

    def __post_init__(self):
        if self.v is None:
            self.v = self.v_leak.copy()
        if self.i_syn is None:
            self.i_syn = np.zeros(self.n_neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()[: self.n_neurons]
        di = (-self.i_syn + self.w_in * x) * (self.dt / self.tau_syn)
        self.i_syn += di
        dv = (self.v_leak - self.v + self.r * self.i_syn) * (self.dt / self.tau_mem)
        self.v += dv
        spikes = (self.v >= self.v_threshold).astype(np.float64)
        self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self):
        self.v = self.v_leak.copy()
        self.i_syn = np.zeros(self.n_neurons)


@dataclass
class SCCubaLINode:
    """Current-based leaky integrator (CubaLIF without threshold).

    NIR CubaLI: tau_syn * dI_syn/dt = -I_syn + w_in * I
                tau_mem * dv/dt = (v_leak - v) + R * I_syn
    """

    name: str
    n_neurons: int
    tau_syn: np.ndarray
    tau_mem: np.ndarray
    r: np.ndarray
    v_leak: np.ndarray
    w_in: np.ndarray
    v: np.ndarray | None = None
    i_syn: np.ndarray | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.CubaLI, dt: float = 1.0) -> SCCubaLINode:
        tau_syn = np.atleast_1d(node.tau_syn).flatten()
        tau_mem = np.atleast_1d(node.tau_mem).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        w_in = np.atleast_1d(node.w_in).flatten()
        return cls(
            name=name,
            n_neurons=len(tau_mem),
            tau_syn=tau_syn,
            tau_mem=tau_mem,
            r=r,
            v_leak=v_leak,
            w_in=w_in,
            dt=dt,
        )

    def __post_init__(self):
        if self.v is None:
            self.v = self.v_leak.copy()
        if self.i_syn is None:
            self.i_syn = np.zeros(self.n_neurons)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(x).flatten()[: self.n_neurons]
        di = (-self.i_syn + self.w_in * x) * (self.dt / self.tau_syn)
        self.i_syn += di
        dv = (self.v_leak - self.v + self.r * self.i_syn) * (self.dt / self.tau_mem)
        self.v += dv
        return self.v.copy()

    def reset(self):
        self.v = self.v_leak.copy()
        self.i_syn = np.zeros(self.n_neurons)


@dataclass
class SCSumPool2dNode:
    """2D sum pooling: sum over spatial kernel windows."""

    name: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]

    @classmethod
    def from_nir(cls, name: str, node: nir.SumPool2d) -> SCSumPool2dNode:
        ks = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
        st = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
        pad = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
        if len(ks) == 1:
            ks = (ks[0], ks[0])
        if len(st) == 1:
            st = (st[0], st[0])
        if len(pad) == 1:
            pad = (pad[0], pad[0])
        return cls(name=name, kernel_size=ks, stride=st, padding=pad)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if x.ndim < 2:
            return x
        # Expect (C, H, W) or (H, W)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        c, h, w = x.shape
        ph, pw = self.padding
        if ph > 0 or pw > 0:
            x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
            h, w = x.shape[1], x.shape[2]
        kh, kw = self.kernel_size
        sh, sw = self.stride
        oh = (h - kh) // sh + 1
        ow = (w - kw) // sw + 1
        out = np.zeros((c, oh, ow))
        for i in range(oh):
            for j in range(ow):
                out[:, i, j] = x[:, i * sh : i * sh + kh, j * sw : j * sw + kw].sum(axis=(1, 2))
        return out.squeeze()


@dataclass
class SCAvgPool2dNode:
    """2D average pooling: SumPool / kernel_area."""

    name: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]

    @classmethod
    def from_nir(cls, name: str, node: nir.AvgPool2d) -> SCAvgPool2dNode:
        ks = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
        st = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
        pad = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
        if len(ks) == 1:
            ks = (ks[0], ks[0])
        if len(st) == 1:
            st = (st[0], st[0])
        if len(pad) == 1:
            pad = (pad[0], pad[0])
        return cls(name=name, kernel_size=ks, stride=st, padding=pad)

    def forward(self, x: np.ndarray) -> np.ndarray:
        sum_node = SCSumPool2dNode(
            name=self.name + "_sum",
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        summed = sum_node.forward(x)
        area = self.kernel_size[0] * self.kernel_size[1]
        return summed / area


@dataclass
class SCConv1dNode:
    """1D convolution: y = conv1d(x, weight) + bias."""

    name: str
    weight: np.ndarray
    bias: np.ndarray
    stride: int
    padding: int
    dilation: int
    groups: int

    @classmethod
    def from_nir(cls, name: str, node: nir.Conv1d) -> SCConv1dNode:
        padding = node.padding if isinstance(node.padding, int) else 0
        return cls(
            name=name,
            weight=node.weight,
            bias=node.bias if node.bias is not None else np.zeros(node.weight.shape[0]),
            stride=node.stride,
            padding=padding,
            dilation=node.dilation,
            groups=node.groups,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (C_in, L) or (L,)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        c_out, c_in_per_group, k = self.weight.shape
        c_in, length = x.shape
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding)), mode="constant")
            length = x.shape[1]
        out_len = (length - self.dilation * (k - 1) - 1) // self.stride + 1
        out = np.zeros((c_out, out_len))
        for o in range(c_out):
            g = o // (c_out // self.groups)
            c_start = g * c_in_per_group
            for l in range(out_len):
                val = 0.0
                for ci in range(c_in_per_group):
                    for ki in range(k):
                        idx = l * self.stride + ki * self.dilation
                        if 0 <= idx < x.shape[1]:
                            val += self.weight[o, ci, ki] * x[c_start + ci, idx]
                out[o, l] = val + self.bias[o]
        return out.squeeze()


@dataclass
class SCConv2dNode:
    """2D convolution: y = conv2d(x, weight) + bias."""

    name: str
    weight: np.ndarray
    bias: np.ndarray
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int

    @classmethod
    def from_nir(cls, name: str, node: nir.Conv2d) -> SCConv2dNode:
        stride = node.stride if isinstance(node.stride, tuple) else (node.stride, node.stride)
        padding = node.padding if isinstance(node.padding, tuple) else (node.padding, node.padding)
        if isinstance(padding[0], str):
            padding = (0, 0)
        dilation = (
            node.dilation if isinstance(node.dilation, tuple) else (node.dilation, node.dilation)
        )
        return cls(
            name=name,
            weight=node.weight,
            bias=node.bias if node.bias is not None else np.zeros(node.weight.shape[0]),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=node.groups,
        )

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: (C_in, H, W) or (H, W)
        if x.ndim == 2:
            x = x[np.newaxis, :, :]
        c_out, c_in_per_group, kh, kw = self.weight.shape
        c_in, h, w = x.shape
        ph, pw = self.padding
        if ph > 0 or pw > 0:
            x = np.pad(x, ((0, 0), (ph, ph), (pw, pw)), mode="constant")
            h, w = x.shape[1], x.shape[2]
        sh, sw = self.stride
        dh, dw = self.dilation
        oh = (h - dh * (kh - 1) - 1) // sh + 1
        ow = (w - dw * (kw - 1) - 1) // sw + 1
        out = np.zeros((c_out, oh, ow))
        for o in range(c_out):
            g = o // (c_out // self.groups)
            c_start = g * c_in_per_group
            for i in range(oh):
                for j in range(ow):
                    val = 0.0
                    for ci in range(c_in_per_group):
                        for ki in range(kh):
                            for kj in range(kw):
                                ii = i * sh + ki * dh
                                jj = j * sw + kj * dw
                                if 0 <= ii < h and 0 <= jj < w:
                                    val += self.weight[o, ci, ki, kj] * x[c_start + ci, ii, jj]
                    out[o, i, j] = val + self.bias[o]
        return out.squeeze()


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
    nir.Delay: lambda name, node, **kw: SCDelayNode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.CubaLIF: lambda name, node, **kw: SCCubaLIFNode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.CubaLI: lambda name, node, **kw: SCCubaLINode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.SumPool2d: lambda name, node, **kw: SCSumPool2dNode.from_nir(name, node),
    nir.AvgPool2d: lambda name, node, **kw: SCAvgPool2dNode.from_nir(name, node),
    nir.Conv1d: lambda name, node, **kw: SCConv1dNode.from_nir(name, node),
    nir.Conv2d: lambda name, node, **kw: SCConv2dNode.from_nir(name, node),
}


def map_node(name: str, node: nir.NIRNode, **kwargs) -> Any:
    """Convert a single NIR node to its SC-NeuroCore equivalent."""
    factory = NODE_MAP.get(type(node))
    if factory is None:
        raise NotImplementedError(
            f"NIR node type {type(node).__name__} not yet supported (node: {name!r})"
        )
    return factory(name, node, **kwargs)
