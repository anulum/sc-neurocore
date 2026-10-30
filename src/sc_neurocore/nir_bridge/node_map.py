# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — NIR node type → SC-NeuroCore primitive factories

"""Execute NIR primitives with explicit Euler clocks and independently owned state.

Stateful voltages use float64 even when source parameters are integer arrays.
The snntorch-subtract convention additionally accounts for the previous
membrane reset signal before spike detection; it differs from SC subtract
when the previous post-reset voltage still exceeds the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index
from typing import Any

import numpy as np

try:
    import nir
except ImportError as e:
    raise ImportError("pip install nir") from e


def _shape_tuple_from_type(type_map: Any, key: str) -> tuple[int, ...] | None:
    """Return a positive integer shape tuple from a NIR type map."""
    if not type_map:
        return None
    raw_shape = type_map.get(key)
    if raw_shape is None:
        return None
    shape = np.atleast_1d(np.asarray(raw_shape, dtype=np.int64)).reshape(-1)
    if shape.size == 0:
        return ()
    if np.any(shape < 0):
        raise ValueError(f"NIR shape for {key!r} contains negative dimensions: {shape}")
    return tuple(int(dim) for dim in shape)


def _shape3_tuple_from_type(type_map: Any, key: str) -> tuple[int, int, int] | None:
    """Return a rank-3 positive integer shape tuple from a NIR type map."""
    shape = _shape_tuple_from_type(type_map, key)
    if shape is None:
        return None
    if len(shape) != 3:
        raise ValueError(f"NIR shape for {key!r} must be rank 3, got {shape}")
    return (shape[0], shape[1], shape[2])


@dataclass
class SCInputNode:
    """Graph entry point — passes input through unchanged."""

    name: str
    shape: tuple[int, ...]

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return the caller-supplied input without changing its values."""
        return x


@dataclass
class SCOutputNode:
    """Graph exit point — collects output."""

    name: str
    shape: tuple[int, ...]
    last_output: np.ndarray[Any, Any] | None = None

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Expose the incoming graph value at this output boundary."""
        self.last_output = x
        return x


@dataclass
class SCLIFNode:
    """LIF neuron mapped from NIR LIF primitive.

    NIR LIF: tau*dv/dt = (v_leak - v) + R*I, spike when v > v_threshold
    Euler: v += ((v_leak - v) + R*I) * dt/tau
    """

    name: str
    n_neurons: int
    tau: np.ndarray[Any, Any]
    r: np.ndarray[Any, Any]
    v_leak: np.ndarray[Any, Any]
    v_threshold: np.ndarray[Any, Any]
    v_reset: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    dt: float = 1.0
    reset_mode: str = "reset"

    @classmethod
    def from_nir(
        cls,
        name: str,
        node: nir.LIF,
        dt: float = 1.0,
        reset_mode: str = "reset",
    ) -> SCLIFNode:
        """Copy LIF parameters and select its Euler timestep and reset convention."""
        tau = np.atleast_1d(node.tau).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = (
            np.atleast_1d(node.v_reset).flatten()
            if getattr(node, "v_reset", None) is not None
            else np.zeros_like(v_threshold)
        )
        return cls(
            name=name,
            n_neurons=len(tau),
            tau=tau,
            r=r,
            v_leak=v_leak,
            v_threshold=v_threshold,
            v_reset=v_reset,
            dt=dt,
            reset_mode=reset_mode,
        )

    def __post_init__(self) -> None:
        """Initialise floating membrane state at the declared leak potential."""
        if self.v is None:
            self.v = self.v_leak.astype(np.float64, copy=True)

    def _broadcast_to(self, size: int) -> None:
        self.n_neurons = size
        for attr in ("tau", "r", "v_leak", "v_threshold", "v_reset"):
            arr = getattr(self, attr)
            if len(arr) == 1 and size > 1:
                setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
        assert self.v is not None
        self.v = np.broadcast_to(self.v, (size,)).copy()

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance Euler voltage, detect spikes and apply the selected reset."""
        x = np.atleast_1d(x).flatten()
        if self.n_neurons == 1 and len(x) > 1:
            self._broadcast_to(len(x))
        x = x[: self.n_neurons]
        prior_reset = (
            (self.v > self.v_threshold) if self.reset_mode == "snntorch-subtract" else None
        )
        dv = (self.v_leak - self.v + self.r * x) * (self.dt / self.tau)
        self.v += dv
        detection = self.v if prior_reset is None else self.v - prior_reset * self.v_threshold
        spikes = (detection > self.v_threshold).astype(np.float64)
        if self.reset_mode in ("subtract", "snntorch-subtract"):
            self.v = np.where(spikes > 0, self.v - self.v_threshold, self.v)
        else:
            self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self) -> None:
        """Restore membrane voltage to the declared leak potential."""
        self.v = self.v_leak.astype(np.float64, copy=True)


@dataclass
class SCIFNode:
    """IF neuron — integrator with threshold, no leak.

    NIR IF: dv/dt = R*I, spike when v > v_threshold
    Euler: v += R*I*dt
    """

    name: str
    n_neurons: int
    r: np.ndarray[Any, Any]
    v_threshold: np.ndarray[Any, Any]
    v_reset: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    dt: float = 1.0
    reset_mode: str = "reset"

    @classmethod
    def from_nir(
        cls,
        name: str,
        node: nir.IF,
        dt: float = 1.0,
        reset_mode: str = "reset",
    ) -> SCIFNode:
        """Copy IF resistance, threshold and reset voltage at the requested timestep."""
        r = np.atleast_1d(node.r).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = (
            np.atleast_1d(node.v_reset).flatten()
            if getattr(node, "v_reset", None) is not None
            else np.zeros_like(r)
        )
        return cls(
            name=name,
            n_neurons=len(r),
            r=r,
            v_threshold=v_threshold,
            v_reset=v_reset,
            dt=dt,
            reset_mode=reset_mode,
        )

    def __post_init__(self) -> None:
        """Initialise the IF membrane at zero in floating precision."""
        if self.v is None:
            self.v = np.zeros(self.n_neurons)

    def _broadcast_to(self, size: int) -> None:
        self.n_neurons = size
        for attr in ("r", "v_threshold", "v_reset"):
            arr = getattr(self, attr)
            if len(arr) == 1 and size > 1:
                setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
        self.v = np.zeros(size)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Integrate input current, detect spikes and apply the selected IF reset."""
        x = np.atleast_1d(x).flatten()
        if self.n_neurons == 1 and len(x) > 1:
            self._broadcast_to(len(x))
        x = x[: self.n_neurons]
        if self.v is None:
            raise RuntimeError("IF state was not initialised")
        prior_reset = (
            (self.v > self.v_threshold) if self.reset_mode == "snntorch-subtract" else None
        )
        self.v += self.r * x * self.dt
        detection = self.v if prior_reset is None else self.v - prior_reset * self.v_threshold
        spikes = (detection > self.v_threshold).astype(np.float64)
        if self.reset_mode in ("subtract", "snntorch-subtract"):
            self.v = np.where(spikes > 0, self.v - self.v_threshold, self.v)
        else:
            self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self) -> None:
        """Clear the IF membrane to its initial zero voltage."""
        self.v = np.zeros(self.n_neurons)


@dataclass
class SCLINode:
    """Leaky integrator — LIF without threshold.

    NIR LI: tau*dv/dt = (v_leak - v) + R*I
    """

    name: str
    n_neurons: int
    tau: np.ndarray[Any, Any]
    r: np.ndarray[Any, Any]
    v_leak: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.LI, dt: float = 1.0) -> SCLINode:
        """Copy the leaky integrator parameters and its Euler timestep."""
        tau = np.atleast_1d(node.tau).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        return cls(name=name, n_neurons=len(tau), tau=tau, r=r, v_leak=v_leak, dt=dt)

    def __post_init__(self) -> None:
        """Initialise floating integrator voltage at the leak potential."""
        if self.v is None:
            self.v = self.v_leak.astype(np.float64, copy=True)

    def _broadcast_to(self, size: int) -> None:
        self.n_neurons = size
        for attr in ("tau", "r", "v_leak"):
            arr = getattr(self, attr)
            if len(arr) == 1 and size > 1:
                setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
        assert self.v is not None
        self.v = np.broadcast_to(self.v, (size,)).copy()

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance the leaky Euler state and return the resulting voltage."""
        assert self.v is not None
        x = np.atleast_1d(x).flatten()
        if self.n_neurons == 1 and len(x) > 1:
            self._broadcast_to(len(x))
        x = x[: self.n_neurons]
        dv = (self.v_leak - self.v + self.r * x) * (self.dt / self.tau)
        self.v += dv
        return self.v.copy()

    def reset(self) -> None:
        """Restore the leaky integrator voltage to its leak potential."""
        self.v = self.v_leak.astype(np.float64, copy=True)


@dataclass
class SCAffineNode:
    """Dense linear transform with bias: y = Wx + b."""

    name: str
    weight: np.ndarray[Any, Any]
    bias: np.ndarray[Any, Any]

    @classmethod
    def from_nir(cls, name: str, node: nir.Affine) -> SCAffineNode:
        """Copy a NIR weight matrix and bias into independent NumPy arrays."""
        return cls(
            name=name, weight=np.asarray(node.weight).copy(), bias=np.asarray(node.bias).copy()
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Flatten the input vector and apply its weighted affine projection."""
        x = np.atleast_1d(x).flatten()
        result: np.ndarray[Any, Any] = self.weight @ x + self.bias
        return result


@dataclass
class SCLinearNode:
    """Matrix multiply without bias: y = Wx."""

    name: str
    weight: np.ndarray[Any, Any]

    @classmethod
    def from_nir(cls, name: str, node: nir.Linear) -> SCLinearNode:
        """Copy a NIR weight matrix into an independent NumPy array."""
        return cls(name=name, weight=np.asarray(node.weight).copy())

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Flatten the input vector and apply its matrix projection."""
        x = np.atleast_1d(x).flatten()
        projected: np.ndarray[Any, Any] = self.weight @ x
        return projected


@dataclass
class SCScaleNode:
    """Element-wise scaling: y = s * x."""

    name: str
    scale: np.ndarray[Any, Any]

    @classmethod
    def from_nir(cls, name: str, node: nir.Scale) -> SCScaleNode:
        """Copy the NIR elementwise scale into an independent NumPy array."""
        return cls(name=name, scale=np.asarray(node.scale).copy())

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Multiply incoming values by the declared elementwise scale."""
        scaled: np.ndarray[Any, Any] = self.scale * x
        return scaled


@dataclass
class SCThresholdNode:
    """Spike threshold: y = 1 if x > threshold else 0."""

    name: str
    threshold: np.ndarray[Any, Any]

    @classmethod
    def from_nir(cls, name: str, node: nir.Threshold) -> SCThresholdNode:
        """Copy the NIR detection thresholds into an independent NumPy array."""
        return cls(name=name, threshold=np.asarray(node.threshold).copy())

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Emit binary spikes where inputs strictly exceed their thresholds."""
        return (x > self.threshold).astype(np.float64)


@dataclass
class SCFlattenNode:
    """Reshape tensor — flatten dimensions."""

    name: str
    start_dim: int
    end_dim: int
    input_shape: tuple[int, ...] | None = None
    output_shape: tuple[int, ...] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.Flatten) -> SCFlattenNode:
        """Retain the source shape and dimension interval of a NIR flatten operation."""
        input_shape = _shape_tuple_from_type(getattr(node, "input_type", None), "input")
        output_shape = _shape_tuple_from_type(getattr(node, "output_type", None), "output")
        return cls(
            name=name,
            start_dim=node.start_dim,
            end_dim=node.end_dim,
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Reshape the declared dimension interval while preserving element order."""
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
    """Pure integrator: dv/dt = R*I (no leak, no threshold). Euler: v += R*I*dt."""

    name: str
    r: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.I, dt: float = 1.0) -> SCIntegratorNode:
        """Copy the NIR integrator resistance and assign its Euler timestep."""
        r = np.atleast_1d(node.r).flatten()
        return cls(name=name, r=r, dt=dt)

    @property
    def n_neurons(self) -> int:
        """Number of integrator state channels."""
        return int(self.r.size)

    def __post_init__(self) -> None:
        """Initialise the integrator state to floating zero."""
        if self.v is None:
            self.v = np.zeros_like(self.r, dtype=np.float64)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Accumulate resistance-weighted input current over one timestep."""
        x = np.atleast_1d(x).flatten()[: len(self.r)]
        if self.v is None:
            raise RuntimeError("integrator state was not initialised")
        self.v += self.r * x * self.dt
        return self.v.copy()

    def reset(self) -> None:
        """Clear the accumulated integrator voltage to floating zero."""
        self.v = np.zeros_like(self.r, dtype=np.float64)


@dataclass
class SCDelayNode:
    """Temporal delay: output = input(t - delay).

    NIR Delay: I(t - tau). Implemented as a circular buffer per element.
    Delay values are rounded to integer timesteps.
    """

    name: str
    delay_steps: np.ndarray[Any, Any]
    delay_time: np.ndarray[Any, Any] | None = None  # original physical time for lossless export
    _buffers: list[list[np.ndarray[Any, Any]]] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.Delay, dt: float = 1.0) -> SCDelayNode:
        """Convert NIR delay times to rounded per-channel timestep counts."""
        delay = np.atleast_1d(node.delay).flatten()
        steps = np.round(delay / dt).astype(int)
        return cls(name=name, delay_steps=steps, delay_time=delay.copy())

    def __post_init__(self) -> None:
        """Allocate the delay history required by the longest channel latency."""
        if self._buffers is None:
            self._buffers = [[np.zeros(1) for _ in range(int(d))] for d in self.delay_steps]

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Return each channel from its declared historical timestep."""
        assert self._buffers is not None
        x = np.atleast_1d(x).flatten()
        out = np.zeros(len(self.delay_steps))
        for i, buf in enumerate(self._buffers):
            xi = float(x[i]) if i < len(x) else 0.0
            if len(buf) == 0:
                out[i] = xi  # zero-delay passthrough
            else:
                out[i] = buf[0][0]
                buf.append(np.array([xi]))
                buf.pop(0)
        return out

    def reset(self) -> None:
        """Clear all channel delay history and restore the buffer index."""
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
    tau_syn: np.ndarray[Any, Any]
    tau_mem: np.ndarray[Any, Any]
    r: np.ndarray[Any, Any]
    v_leak: np.ndarray[Any, Any]
    v_threshold: np.ndarray[Any, Any]
    v_reset: np.ndarray[Any, Any]
    w_in: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    i_syn: np.ndarray[Any, Any] | None = None
    dt: float = 1.0
    reset_mode: str = "reset"

    @classmethod
    def from_nir(
        cls,
        name: str,
        node: nir.CubaLIF,
        dt: float = 1.0,
        reset_mode: str = "reset",
    ) -> SCCubaLIFNode:
        """Copy current-based LIF parameters, timestep and reset convention."""
        tau_syn = np.atleast_1d(node.tau_syn).flatten()
        tau_mem = np.atleast_1d(node.tau_mem).flatten()
        r = np.atleast_1d(node.r).flatten()
        v_leak = np.atleast_1d(node.v_leak).flatten()
        v_threshold = np.atleast_1d(node.v_threshold).flatten()
        v_reset = (
            np.atleast_1d(node.v_reset).flatten()
            if getattr(node, "v_reset", None) is not None
            else np.zeros_like(v_threshold)
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
            reset_mode=reset_mode,
        )

    def __post_init__(self) -> None:
        """Initialise floating membrane voltage and zero synaptic current."""
        if self.v is None:
            self.v = self.v_leak.astype(np.float64, copy=True)
        if self.i_syn is None:
            self.i_syn = np.zeros(self.n_neurons)

    def _broadcast_to(self, size: int) -> None:
        """Broadcast scalar params to match actual input size."""
        self.n_neurons = size
        for attr in ("tau_syn", "tau_mem", "r", "v_leak", "v_threshold", "v_reset", "w_in"):
            arr = getattr(self, attr)
            if len(arr) == 1 and size > 1:
                setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
        assert self.v is not None
        self.v = np.broadcast_to(self.v, (size,)).copy()
        self.i_syn = np.zeros(size)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance synaptic current before membrane voltage, then detect and reset spikes."""
        assert self.v is not None and self.i_syn is not None
        x = np.atleast_1d(x).flatten()
        if self.n_neurons == 1 and len(x) > 1:
            self._broadcast_to(len(x))
        x = x[: self.n_neurons]
        di = (-self.i_syn + self.w_in * x) * (self.dt / self.tau_syn)
        self.i_syn += di
        prior_reset = (
            (self.v > self.v_threshold) if self.reset_mode == "snntorch-subtract" else None
        )
        dv = (self.v_leak - self.v + self.r * self.i_syn) * (self.dt / self.tau_mem)
        self.v += dv
        detection = self.v if prior_reset is None else self.v - prior_reset * self.v_threshold
        spikes = (detection > self.v_threshold).astype(np.float64)
        if self.reset_mode in ("subtract", "snntorch-subtract"):
            self.v = np.where(spikes > 0, self.v - self.v_threshold, self.v)
        else:
            self.v = np.where(spikes > 0, self.v_reset, self.v)
        return spikes

    def reset(self) -> None:
        """Restore leak voltage and clear the accumulated synaptic current."""
        self.v = self.v_leak.astype(np.float64, copy=True)
        self.i_syn = np.zeros(self.n_neurons)


@dataclass
class SCCubaLINode:
    """Current-based leaky integrator (CubaLIF without threshold).

    NIR CubaLI: tau_syn * dI_syn/dt = -I_syn + w_in * I
                tau_mem * dv/dt = (v_leak - v) + R * I_syn
    """

    name: str
    n_neurons: int
    tau_syn: np.ndarray[Any, Any]
    tau_mem: np.ndarray[Any, Any]
    r: np.ndarray[Any, Any]
    v_leak: np.ndarray[Any, Any]
    w_in: np.ndarray[Any, Any]
    v: np.ndarray[Any, Any] | None = None
    i_syn: np.ndarray[Any, Any] | None = None
    dt: float = 1.0

    @classmethod
    def from_nir(cls, name: str, node: nir.CubaLI, dt: float = 1.0) -> SCCubaLINode:
        """Copy current-based integrator parameters and its Euler timestep."""
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

    def __post_init__(self) -> None:
        """Initialise the current-based integrator voltage and synaptic current."""
        if self.v is None:
            self.v = self.v_leak.astype(np.float64, copy=True)
        if self.i_syn is None:
            self.i_syn = np.zeros(self.n_neurons)

    def _broadcast_to(self, size: int) -> None:
        self.n_neurons = size
        for attr in ("tau_syn", "tau_mem", "r", "v_leak", "w_in"):
            arr = getattr(self, attr)
            if len(arr) == 1 and size > 1:
                setattr(self, attr, np.broadcast_to(arr, (size,)).copy())
        assert self.v is not None
        self.v = np.broadcast_to(self.v, (size,)).copy()
        self.i_syn = np.zeros(size)

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Advance the synaptic and membrane Euler states and return voltage."""
        assert self.v is not None and self.i_syn is not None
        x = np.atleast_1d(x).flatten()
        if self.n_neurons == 1 and len(x) > 1:
            self._broadcast_to(len(x))
        x = x[: self.n_neurons]
        di = (-self.i_syn + self.w_in * x) * (self.dt / self.tau_syn)
        self.i_syn += di
        dv = (self.v_leak - self.v + self.r * self.i_syn) * (self.dt / self.tau_mem)
        self.v += dv
        return self.v.copy()

    def reset(self) -> None:
        """Restore leak voltage and clear the current-based integrator current."""
        self.v = self.v_leak.astype(np.float64, copy=True)
        self.i_syn = np.zeros(self.n_neurons)


@dataclass
class SCSumPool2dNode:
    """2D sum pooling: sum over spatial kernel windows."""

    name: str
    kernel_size: tuple[int, int]
    stride: tuple[int, int]
    padding: tuple[int, int]
    input_shape: tuple[int, int, int] | None = None
    output_shape: tuple[int, int, int] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.SumPool2d) -> SCSumPool2dNode:
        """Retain sum-pooling geometry and declared channel-spatial input shape."""
        ks_raw = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
        st_raw = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
        pad_raw = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
        ks = (ks_raw[0], ks_raw[0]) if len(ks_raw) == 1 else (ks_raw[0], ks_raw[1])
        st = (st_raw[0], st_raw[0]) if len(st_raw) == 1 else (st_raw[0], st_raw[1])
        pad = (pad_raw[0], pad_raw[0]) if len(pad_raw) == 1 else (pad_raw[0], pad_raw[1])
        return cls(
            name=name,
            kernel_size=ks,
            stride=st,
            padding=pad,
            input_shape=_shape3_tuple_from_type(getattr(node, "input_type", None), "input"),
            output_shape=_shape3_tuple_from_type(getattr(node, "output_type", None), "output"),
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Sum spatial windows with the declared padding, stride and kernel."""
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
    input_shape: tuple[int, int, int] | None = None
    output_shape: tuple[int, int, int] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.AvgPool2d) -> SCAvgPool2dNode:
        """Retain average-pooling geometry and declared channel-spatial input shape."""
        ks_raw = tuple(int(x) for x in np.atleast_1d(node.kernel_size).flatten()[:2])
        st_raw = tuple(int(x) for x in np.atleast_1d(node.stride).flatten()[:2])
        pad_raw = tuple(int(x) for x in np.atleast_1d(node.padding).flatten()[:2])
        ks = (ks_raw[0], ks_raw[0]) if len(ks_raw) == 1 else (ks_raw[0], ks_raw[1])
        st = (st_raw[0], st_raw[0]) if len(st_raw) == 1 else (st_raw[0], st_raw[1])
        pad = (pad_raw[0], pad_raw[0]) if len(pad_raw) == 1 else (pad_raw[0], pad_raw[1])
        return cls(
            name=name,
            kernel_size=ks,
            stride=st,
            padding=pad,
            input_shape=_shape3_tuple_from_type(getattr(node, "input_type", None), "input"),
            output_shape=_shape3_tuple_from_type(getattr(node, "output_type", None), "output"),
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Average spatial windows with the declared padding, stride and kernel."""
        sum_node = SCSumPool2dNode(
            name=self.name + "_sum",
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        summed = sum_node.forward(x)
        area = self.kernel_size[0] * self.kernel_size[1]
        return summed / area


def _conv_integer(value: Any, field_name: str, minimum: int = 1) -> int:
    """Own one integral convolution extent without truncating fractional geometry."""
    try:
        scalar = np.asarray(value).item()
        if isinstance(scalar, bool):
            raise TypeError
        result = index(scalar)
    except (TypeError, ValueError):
        raise ValueError(f"Convolution {field_name} must be a single integer") from None
    if result < minimum:
        raise ValueError(f"Convolution {field_name} must be at least {minimum}")
    return result


def _resolve_conv_padding(spec: int | str, *, kernel: int, dilation: int, stride: int) -> int:
    """Resolve a NIR convolution padding spec to a symmetric integer pad per side.

    Integer specs pass through. ``"valid"`` maps to 0. ``"same"`` returns the
    symmetric padding that preserves the spatial size; it requires stride 1 and an
    even effective kernel span (``dilation * (kernel - 1)``), which holds for every
    odd kernel. Even-kernel ``"same"`` would need asymmetric padding that this
    symmetric path cannot represent, so it is rejected with an explicit message.
    """
    if isinstance(spec, str):
        mode = spec.lower()
        if mode == "valid":
            return 0
        if mode == "same":
            if stride != 1:
                raise ValueError("Conv 'same' padding requires stride 1")
            span = dilation * (kernel - 1)
            if span % 2 != 0:
                raise ValueError(
                    "Conv 'same' padding needs an even effective kernel span "
                    "(dilation * (kernel - 1)); use explicit integer padding for "
                    "this kernel/dilation"
                )
            return span // 2
        raise ValueError(
            f"Unsupported conv padding mode {spec!r}; use 'same', 'valid', or an integer"
        )
    return _conv_integer(spec, "padding", minimum=0)


@dataclass
class SCConv1dNode:
    """1D convolution: y = conv1d(x, weight) + bias."""

    name: str
    weight: np.ndarray[Any, Any]
    bias: np.ndarray[Any, Any]
    stride: int
    padding: int
    dilation: int
    groups: int
    input_shape: int | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.Conv1d) -> SCConv1dNode:
        """Copy one-dimensional convolution weights and resolve symmetric padding."""
        stride = _conv_integer(node.stride, "stride")
        dilation = _conv_integer(node.dilation, "dilation")
        padding = _resolve_conv_padding(
            node.padding,
            kernel=int(node.weight.shape[2]),
            dilation=dilation,
            stride=stride,
        )
        raw_shape = getattr(node, "input_shape", None)
        input_shape = None if raw_shape is None else _conv_integer(raw_shape, "input_shape")
        return cls(
            name=name,
            weight=np.asarray(node.weight).copy(),
            bias=np.asarray(node.bias).copy()
            if node.bias is not None
            else np.zeros(node.weight.shape[0]),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=_conv_integer(node.groups, "groups"),
            input_shape=input_shape,
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # x: (C_in, L) or (L,)
        """Apply the grouped one-dimensional convolution to channel-spatial input."""
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
    weight: np.ndarray[Any, Any]
    bias: np.ndarray[Any, Any]
    stride: tuple[int, int]
    padding: tuple[int, int]
    dilation: tuple[int, int]
    groups: int
    input_shape: tuple[int, int] | None = None
    output_shape: tuple[int, int, int] | None = None

    @classmethod
    def from_nir(cls, name: str, node: nir.Conv2d) -> SCConv2dNode:
        """Copy two-dimensional convolution weights and normalise spatial geometry."""
        raw_stride = np.broadcast_to(np.asarray(node.stride), (2,))
        stride = (_conv_integer(raw_stride[0], "stride"), _conv_integer(raw_stride[1], "stride"))
        raw_dilation = np.broadcast_to(np.asarray(node.dilation), (2,))
        dilation = (
            _conv_integer(raw_dilation[0], "dilation"),
            _conv_integer(raw_dilation[1], "dilation"),
        )
        kh, kw = int(node.weight.shape[2]), int(node.weight.shape[3])
        raw_padding = tuple(np.broadcast_to(np.asarray(node.padding), (2,)))
        padding = (
            _resolve_conv_padding(
                raw_padding[0], kernel=kh, dilation=dilation[0], stride=stride[0]
            ),
            _resolve_conv_padding(
                raw_padding[1], kernel=kw, dilation=dilation[1], stride=stride[1]
            ),
        )
        raw_shape = getattr(node, "input_shape", None)
        input_shape = None
        if raw_shape is not None:
            height, width = raw_shape
            input_shape = (
                _conv_integer(height, "input_shape"),
                _conv_integer(width, "input_shape"),
            )
        output_shape = _shape3_tuple_from_type(getattr(node, "output_type", None), "output")
        if input_shape is not None and output_shape is not None:
            height, width = (int(v) for v in input_shape)
            expected = (
                int(node.weight.shape[0]),
                (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1,
                (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1,
            )
            if output_shape != expected:
                raise ValueError(
                    "Conv2d output shape conflicts with its kernel geometry; "
                    "this NIR version infers non-square kernels incorrectly"
                )
        return cls(
            name=name,
            weight=np.asarray(node.weight).copy(),
            bias=np.asarray(node.bias).copy()
            if node.bias is not None
            else np.zeros(node.weight.shape[0]),
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=_conv_integer(node.groups, "groups"),
            input_shape=input_shape,
            output_shape=output_shape,
        )

    def forward(self, x: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        # x: (C_in, H, W) or (H, W)
        """Apply the grouped two-dimensional convolution to channel-spatial input."""
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
    nir.LIF: lambda name, node, **kw: SCLIFNode.from_nir(
        name, node, dt=kw.get("dt", 1.0), reset_mode=kw.get("reset_mode", "reset")
    ),
    nir.IF: lambda name, node, **kw: SCIFNode.from_nir(
        name, node, dt=kw.get("dt", 1.0), reset_mode=kw.get("reset_mode", "reset")
    ),
    nir.LI: lambda name, node, **kw: SCLINode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.I: lambda name, node, **kw: SCIntegratorNode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.Affine: lambda name, node, **kw: SCAffineNode.from_nir(name, node),
    nir.Linear: lambda name, node, **kw: SCLinearNode.from_nir(name, node),
    nir.Scale: lambda name, node, **kw: SCScaleNode.from_nir(name, node),
    nir.Threshold: lambda name, node, **kw: SCThresholdNode.from_nir(name, node),
    nir.Flatten: lambda name, node, **kw: SCFlattenNode.from_nir(name, node),
    nir.Delay: lambda name, node, **kw: SCDelayNode.from_nir(name, node, dt=kw.get("dt", 1.0)),
    nir.CubaLIF: lambda name, node, **kw: SCCubaLIFNode.from_nir(
        name, node, dt=kw.get("dt", 1.0), reset_mode=kw.get("reset_mode", "reset")
    ),
    nir.SumPool2d: lambda name, node, **kw: SCSumPool2dNode.from_nir(name, node),
    nir.AvgPool2d: lambda name, node, **kw: SCAvgPool2dNode.from_nir(name, node),
    nir.Conv1d: lambda name, node, **kw: SCConv1dNode.from_nir(name, node),
    nir.Conv2d: lambda name, node, **kw: SCConv2dNode.from_nir(name, node),
}


if hasattr(nir, "CubaLI"):
    NODE_MAP[nir.CubaLI] = lambda name, node, **kw: SCCubaLINode.from_nir(
        name, node, dt=kw.get("dt", 1.0)
    )


def map_node(name: str, node: nir.NIRNode, **kwargs: Any) -> Any:
    """Convert a single NIR node to its SC-NeuroCore equivalent."""
    factory = NODE_MAP.get(type(node))
    if factory is None:
        raise NotImplementedError(
            f"NIR node type {type(node).__name__} not yet supported (node: {name!r})"
        )
    return factory(name, node, **kwargs)
