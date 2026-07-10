# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Indiveri et al. 2011 — DYNAP-SE differential-pair integrator

from __future__ import annotations

import importlib as _importlib
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

try:
    _EngineCls: Optional[type[Any]] = _importlib.import_module(
        "sc_neurocore_engine"
    ).DPINeuron
    _HAS_RUST = True
except (ImportError, AttributeError):
    _EngineCls = None
    _HAS_RUST = False

_RUST_ENGINE_DEFAULTS: dict[str, float] = {
    "i_mem": 0.0,
    "i_threshold": 1.0,
    "i_reset": 0.0,
    "i_leak": 0.01,
    "tau": 20.0,
    "gain": 1.0,
    "dt": 1.0,
}


@dataclass
class DPINeuron:
    """Indiveri et al. 2011 — DYNAP-SE differential-pair integrator.

    Subthreshold log-domain dynamics modelling analog VLSI circuits.
    tau dI_mem/dt = -I_mem + I_syn + I_leak
    Spike when I_mem >= I_threshold, reset to I_reset.
    All variables in current domain (nA), mirroring transistor currents.

    Reference: Chicca, E. et al. (2014). Proc. IEEE 102:1367–1388.

    ``simulate`` supports ``backend`` values ``python``, ``rust``, and ``auto``.
    """

    i_mem: float = 0.0
    i_threshold: float = 1.0
    i_reset: float = 0.0
    i_leak: float = 0.01
    tau: float = 20.0
    gain: float = 1.0
    dt: float = 1.0

    def _matches_rust_engine_contract(self) -> bool:
        """Return whether the instance matches the Rust engine default contract."""
        for name, expected in _RUST_ENGINE_DEFAULTS.items():
            if float(getattr(self, name)) != expected:
                return False
        return True

    def step(self, i_syn: float) -> int:
        if not math.isfinite(i_syn):
            raise ValueError("i_syn must be finite")
        di = (-self.i_mem + self.gain * i_syn + self.i_leak) / self.tau * self.dt
        self.i_mem += di
        self.i_mem = max(self.i_mem, 0.0)
        if self.i_mem >= self.i_threshold:
            self.i_mem = self.i_reset
            return 1
        return 0

    def simulate(
        self, n_steps: int, current: float = 0.0, backend: str = "auto"
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Advance ``n_steps`` updates, returning ``(i_mem_trace, spikes)``.

        ``current`` is the synaptic input current ``I_syn`` (constant drive).
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if backend not in ("auto", "python", "rust"):
            raise ValueError(f"backend must be auto/python/rust, got {backend!r}")
        if not math.isfinite(current):
            raise ValueError("current must be finite")
        prefer_rust = backend == "rust" or (
            backend == "auto" and _HAS_RUST and self._matches_rust_engine_contract()
        )
        if prefer_rust:
            if not _HAS_RUST or _EngineCls is None:
                raise RuntimeError(
                    "Rust DPI backend requested but sc_neurocore_engine is unavailable."
                )
            if not self._matches_rust_engine_contract():
                raise RuntimeError(
                    "Rust DPI backend requires factory-default parameters and initial state."
                )
            trace, spikes, state = self._simulate_rust(n_steps, current)
        else:
            if backend == "rust":
                raise RuntimeError(
                    "Rust DPI backend requested but sc_neurocore_engine is unavailable."
                )
            trace, spikes, state = self._simulate_python(n_steps, current)
        self.i_mem = state
        return trace, spikes

    def _simulate_python(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += self.step(current)
            trace[t] = self.i_mem
        return trace, spikes, self.i_mem

    def _simulate_rust(
        self, n_steps: int, current: float
    ) -> tuple[npt.NDArray[np.float64], int, float]:
        assert _EngineCls is not None
        neuron = _EngineCls()
        trace = np.empty(n_steps, dtype=np.float64)
        spikes = 0
        for t in range(n_steps):
            spikes += int(neuron.step(float(current)))
            trace[t] = float(neuron.get_state()["i_mem"])
        return trace, spikes, float(neuron.get_state()["i_mem"])

    def reset(self) -> None:
        self.i_mem = 0.0
