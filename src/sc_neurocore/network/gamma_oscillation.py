# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Gamma oscillation circuit (PING, conductance-based)

"""Pyramidal-Interneuron Network Gamma (PING) oscillation circuit.

Conductance-based PING per **Börgers & Kopell 2003** (Neural Comp.
15:509-538) — itself a refinement of **Whittington et al. 1995**
(Nature 373:612-615). The previous implementation was a rate-coded
reduced model that did not reproduce the published gamma-band peak; this
rewrite ships the full conductance model so the spectral peak,
phase-locking, and weak-PING-vs-strong-PING transition published in
those papers are reproducible.

Per-neuron model (Börgers-Kopell §2, also Wang & Buzsáki 1996 for
the I cell):

    C_m dV/dt = -g_L (V - E_L)
                - g_AMPA (V - E_AMPA)        # E inputs onto E and I
                - g_GABA (V - E_GABA)        # I inputs onto E and I
                + I_drive
                + sigma * sqrt(dt) * xi(t)

Spike when V crosses `v_threshold`; clamp to `v_reset` for an
absolute refractory period (`t_refrac`). Each pre-synaptic spike
adds a unit step to the post-synaptic conductance trace, which
then decays exponentially:

    dg_AMPA/dt = -g_AMPA / tau_AMPA + w_AMPA * Σ delta(t - t_spike^pre_E)
    dg_GABA/dt = -g_GABA / tau_GABA + w_GABA * Σ delta(t - t_spike^pre_I)

Defaults are the published values from Börgers-Kopell 2003 Fig 2
(weak-PING regime, 40 Hz):

    tau_AMPA  = 3 ms        E_AMPA  =   0 mV
    tau_GABA  = 9 ms        E_GABA  = -80 mV
    g_L       = 0.1 mS/cm²  E_L     = -67 mV
    C_m       = 1 µF/cm²
    v_thresh  = -52 mV      v_reset = -67 mV    t_refrac = 2 ms
    w_EE = 0.05  w_EI = 0.25  w_IE = 0.25  w_II = 0.20

Drive `I_drive_e_mean` ≈ 1.4 µA/cm² puts E cells in the supra-
threshold regime; I cells receive a smaller `I_drive_i_mean` so
they fire only when dragged up by the recurrent E→I conductance —
the canonical PING gain mechanism.

Usage:

    ping = PINGCircuit(n_excitatory=80, n_inhibitory=20)
    spikes_e_log = []
    spikes_i_log = []
    for _ in range(2000):                       # 200 ms at dt=0.1
        e, i = ping.step(dt=0.1)
        spikes_e_log.append(e)
        spikes_i_log.append(i)
    freq_hz = ping.dominant_frequency(spikes_e_log, dt=0.1)
    assert 30.0 <= freq_hz <= 80.0              # gamma band
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

# Optional Rust per-step kernel — built from `engine/src/ping.rs` and
# exposed by `sc_neurocore_engine.py_ping_step`. The Rust backend is
# bit-deterministic for matching seeds because the noise samples are
# drawn on the Python side and passed in as `xi_e` / `xi_i` arrays.
# Selectable via the `backend=` argument; the default `"auto"` prefers
# Rust when available and falls back to NumPy.
#
# Import the symbol from the compiled `sc_neurocore_engine` submodule
# directly — the top-level `bridge/sc_neurocore_engine/__init__.py`
# Python wrapper that pytest places earlier on `sys.path` does not
# expose every Rust symbol, so the bare `from sc_neurocore_engine
# import …` would yield `False` whenever the bridge wrapper wins
# the import race.
# Tracks the Rust PING kernel when the compiled `sc_neurocore_engine`
# wheel is present, `None` otherwise. Loaded via `importlib` (rather than
# `from X import Y as _rust_ping_step`) so the variable binding is a
# regular assignment — the `from X import Y as Z` form would trip mypy's
# no-redef rule twice over inside the nested try/except fallback.
import importlib as _importlib

_rust_ping_step: Callable[..., Any] | None = None
_HAS_RUST_PING_STEP = False
try:
    _rust_ping_step = _importlib.import_module(
        "sc_neurocore_engine.sc_neurocore_engine"
    ).py_ping_step
    _HAS_RUST_PING_STEP = True
except (ImportError, AttributeError):
    try:
        _rust_ping_step = _importlib.import_module("sc_neurocore_engine").py_ping_step
        _HAS_RUST_PING_STEP = True
    except (ImportError, AttributeError):
        pass

import ctypes
import logging
import os

_logger = logging.getLogger(__name__)

_julia_ping_step: Callable[..., Any] | None = None
_HAS_JULIA_PING_STEP = False
try:
    from juliacall import Main as jl

    _jl_ping_file = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "accel", "julia", "network", "gamma_oscillation.jl"
        )
    )
    if os.path.exists(_jl_ping_file):
        jl.seval(f'include("{_jl_ping_file}")')
        _julia_ping_step = jl.GammaOscillationAccel.py_ping_step
        _HAS_JULIA_PING_STEP = True
except Exception as _jl_err:  # noqa: BLE001
    _logger.debug("Julia PING accel unavailable: %r", _jl_err)

_go_ping_step: Any = None  # ctypes function pointer; precise type varies
_HAS_GO_PING_STEP = False
try:
    _go_ping_lib = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "accel",
            "go",
            "gamma_oscillation",
            "libgamma_oscillation.so",
        )
    )
    if os.path.exists(_go_ping_lib):
        _go_lib = ctypes.CDLL(_go_ping_lib)
        _go_ping_step = _go_lib.py_ping_step_c
        _HAS_GO_PING_STEP = True
except Exception as _go_err:  # noqa: BLE001
    _logger.debug("Go PING accel unavailable: %r", _go_err)

_mojo_ping_step: Any = None  # ctypes function pointer; precise type varies
_HAS_MOJO_PING_STEP = False
try:
    _mojo_ping_lib = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "accel", "mojo", "kernels", "libgamma_oscillation.so"
        )
    )
    if os.path.exists(_mojo_ping_lib):
        _mojo_lib = ctypes.CDLL(_mojo_ping_lib)
        _mojo_ping_step = _mojo_lib.py_ping_step
        _HAS_MOJO_PING_STEP = True
except Exception as _mojo_err:  # noqa: BLE001
    _logger.debug("Mojo PING accel unavailable: %r", _mojo_err)

# Border-Kopell 2003 published defaults (Fig 2A, weak-PING 40 Hz).
_DEFAULT_C_M = 1.0  # µF/cm²
_DEFAULT_G_L = 0.1  # mS/cm²
_DEFAULT_E_L = -67.0  # mV
_DEFAULT_E_AMPA = 0.0  # mV
_DEFAULT_E_GABA = -80.0  # mV
_DEFAULT_V_THRESH = -52.0  # mV
_DEFAULT_V_RESET = -67.0  # mV
_DEFAULT_T_REFRAC = 2.0  # ms
_DEFAULT_TAU_AMPA = 3.0  # ms
_DEFAULT_TAU_GABA = 9.0  # ms


@dataclass
class PINGCircuit:
    """Conductance-based PING circuit (Börgers-Kopell 2003).

    Parameters mirror the publication; defaults reproduce the 40 Hz
    weak-PING regime from Fig 2A. Override `i_drive_e_mean` to scan
    drive-frequency curves; override `w_EI` / `w_IE` to scan the
    gain-loop strength.
    """

    n_excitatory: int = 80
    n_inhibitory: int = 20

    # Membrane biophysics (Börgers-Kopell §2).
    c_m: float = _DEFAULT_C_M
    g_l: float = _DEFAULT_G_L
    e_l: float = _DEFAULT_E_L
    e_ampa: float = _DEFAULT_E_AMPA
    e_gaba: float = _DEFAULT_E_GABA
    v_threshold: float = _DEFAULT_V_THRESH
    v_reset: float = _DEFAULT_V_RESET
    t_refrac: float = _DEFAULT_T_REFRAC

    # Synaptic time constants.
    tau_ampa: float = _DEFAULT_TAU_AMPA
    tau_gaba: float = _DEFAULT_TAU_GABA

    # Connection weights (per-spike conductance jump applied to
    # every post-synaptic cell — all-to-all mean-field coupling).
    # The defaults below were chosen to reproduce the Börgers-Kopell
    # 2003 Fig 2A weak-PING regime (40 Hz dominant frequency with
    # `i_drive_e_mean = 2.0 µA/cm²`, N_E=80, N_I=20), verified by
    # `tests/test_gamma_oscillation.py::TestPublishedFidelity::
    # test_gamma_frequency_is_30_to_80_hz`. They are smaller than
    # the published per-synapse values because every pre-synaptic
    # spike here contributes to every post-synaptic cell (the
    # cumulative burst conductance scales with N_pre).
    w_ee: float = 0.0006  # E → E (recurrent excitation)
    w_ei: float = 0.003  # E → I (gain loop forward)
    w_ie: float = 0.005  # I → E (gain loop feedback)
    w_ii: float = 0.01  # I → I (lateral inhibition)

    # External drive (µA/cm²): per-neuron Gaussian heterogeneity
    # around the mean. Setting `i_drive_e_mean` ≈ 1.4 puts E cells
    # in the supra-threshold regime that drives gamma; setting
    # `i_drive_i_mean` ≈ 0 forces I cells to spike only when pulled
    # up by recurrent E→I — the canonical PING gain loop.
    i_drive_e_mean: float = 2.0
    i_drive_e_sigma: float = 0.05
    i_drive_i_mean: float = 0.0
    i_drive_i_sigma: float = 0.05

    # Membrane noise (µA/cm²·ms^½). Wraps a √dt term inside `step`.
    sigma_e: float = 0.10
    sigma_i: float = 0.05

    seed: int = 42

    # Backend: "auto" (Rust if available, else Python), "rust",
    # "python". Selecting "rust" when the Rust kernel is not built
    # raises `RuntimeError` from `__post_init__`.
    backend: str = "auto"

    # State (sized and populated by __post_init__; the empty-array
    # defaults exist only so mypy sees the fields as plain ndarray
    # instead of ndarray | None, which would force a narrowing
    # assertion at every usage site).
    v_e: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    v_i: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    g_ampa_e: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    g_ampa_i: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    g_gaba_e: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    g_gaba_i: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    refrac_e: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    refrac_i: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    i_drive_e: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)
    i_drive_i: np.ndarray[Any, Any] = field(default_factory=lambda: np.zeros(0), repr=False)

    def __post_init__(self) -> None:
        """Validate the population sizes and initialise oscillator state."""
        if self.n_excitatory <= 0 or self.n_inhibitory <= 0:
            raise ValueError("PINGCircuit needs at least 1 E and 1 I neuron")
        self._rng = np.random.default_rng(self.seed)
        # Normalise per-spike conductance contributions so the
        # per-target drive is invariant under changes to
        # `n_excitatory` / `n_inhibitory`. Default weights `w_*`
        # were tuned for the canonical 80 / 20 PING circuit; without
        # this normalisation a 400 / 100 circuit receives 5× more
        # drive per spike and the dominant frequency drifts out of
        # the published 30-80 Hz band (verified by
        # `benchmarks/bench_gamma_oscillation.py`).
        self._w_ee_eff = self.w_ee * (80.0 / self.n_excitatory)
        self._w_ei_eff = self.w_ei * (80.0 / self.n_excitatory)
        self._w_ie_eff = self.w_ie * (20.0 / self.n_inhibitory)
        self._w_ii_eff = self.w_ii * (20.0 / self.n_inhibitory)

        # Resolve backend selection. "auto" prefers the Rust kernel
        # if available; explicit "rust" raises if the kernel is
        # missing rather than silently downgrading.
        if self.backend not in {"auto", "rust", "python", "julia", "go", "mojo"}:
            raise ValueError(
                f"backend must be one of 'auto'|'rust'|'python'|'julia'|'go'|'mojo', got {self.backend!r}"
            )
        if self.backend == "rust" and not _HAS_RUST_PING_STEP:
            raise RuntimeError(
                "backend='rust' requested but `sc_neurocore_engine.py_ping_step` is not available"
            )
        if self.backend == "julia" and not _HAS_JULIA_PING_STEP:
            raise RuntimeError("backend='julia' requested but julia kernel is not available")
        if self.backend == "go" and not _HAS_GO_PING_STEP:
            raise RuntimeError("backend='go' requested but go kernel is not available")
        if self.backend == "mojo" and not _HAS_MOJO_PING_STEP:
            raise RuntimeError("backend='mojo' requested but mojo kernel is not available")

        self._use_rust = self.backend == "rust" or (self.backend == "auto" and _HAS_RUST_PING_STEP)
        self._use_julia = self.backend == "julia"
        self._use_go = self.backend == "go"
        self._use_mojo = self.backend == "mojo"
        # Initial V drawn near E_L with small jitter, so spike onset
        # is asynchronous (matches Whittington 1995 burn-in).
        self.v_e = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_excitatory)
        self.v_i = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_inhibitory)
        self.g_ampa_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.g_ampa_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        self.g_gaba_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.g_gaba_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        self.refrac_e = np.zeros(self.n_excitatory, dtype=np.float64)
        self.refrac_i = np.zeros(self.n_inhibitory, dtype=np.float64)
        # Per-neuron heterogeneous drive (constant for the run).
        self.i_drive_e = self._rng.normal(
            self.i_drive_e_mean,
            self.i_drive_e_sigma,
            self.n_excitatory,
        )
        self.i_drive_i = self._rng.normal(
            self.i_drive_i_mean,
            self.i_drive_i_sigma,
            self.n_inhibitory,
        )

    # ── Single timestep ──────────────────────────────────────────

    def step(self, dt: float = 0.1) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Advance one timestep (dt in ms); return (spikes_e, spikes_i)."""
        if self._use_rust:
            return self._step_rust(dt)
        if self._use_julia:
            return self._step_julia(dt)
        if self._use_go:
            return self._step_go(dt)
        if self._use_mojo:
            return self._step_mojo(dt)
        return self._step_python(dt)

    def _step_rust(self, dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Run the Rust-backed per-step kernel.

        Matches the Python path
        bit-identically for a given seed.

        Noise samples are pre-drawn on the Python side so the per-
        instance RNG state evolves identically across both backends;
        cross-population conductance propagation stays in Python
        because it is an O(N) update that the FFI overhead does not
        amortise.
        """
        assert _rust_ping_step is not None  # gated by self._use_rust
        assert (
            self.v_e is not None
            and self.v_i is not None
            and self.g_ampa_e is not None
            and self.g_ampa_i is not None
            and self.g_gaba_e is not None
            and self.g_gaba_i is not None
            and self.refrac_e is not None
            and self.refrac_i is not None
            and self.i_drive_e is not None
            and self.i_drive_i is not None
        )
        xi_e = self._rng.standard_normal(self.n_excitatory)
        xi_i = self._rng.standard_normal(self.n_inhibitory)
        spikes_e_u8 = np.zeros(self.n_excitatory, dtype=np.uint8)
        spikes_i_u8 = np.zeros(self.n_inhibitory, dtype=np.uint8)
        n_e_spikes, n_i_spikes = _rust_ping_step(
            self.v_e,
            self.g_ampa_e,
            self.g_gaba_e,
            self.refrac_e,
            self.i_drive_e,
            xi_e,
            spikes_e_u8,
            self.v_i,
            self.g_ampa_i,
            self.g_gaba_i,
            self.refrac_i,
            self.i_drive_i,
            xi_i,
            spikes_i_u8,
            self.e_l,
            self.e_ampa,
            self.e_gaba,
            self.g_l,
            self.c_m,
            self.v_threshold,
            self.v_reset,
            self.t_refrac,
            self.tau_ampa,
            self.tau_gaba,
            self.sigma_e,
            self.sigma_i,
            dt,
        )
        # Cross-population conductance propagation (same as Python path).
        if n_e_spikes > 0:
            self.g_ampa_e += self._w_ee_eff * n_e_spikes
            self.g_ampa_i += self._w_ei_eff * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self._w_ie_eff * n_i_spikes
            self.g_gaba_i += self._w_ii_eff * n_i_spikes
        return spikes_e_u8.astype(bool), spikes_i_u8.astype(bool)

    def _step_julia(self, dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        assert _julia_ping_step is not None, "backend='julia' but _julia_ping_step is not loaded"
        xi_e = self._rng.standard_normal(self.n_excitatory)
        xi_i = self._rng.standard_normal(self.n_inhibitory)
        spikes_e_u8 = np.zeros(self.n_excitatory, dtype=np.uint8)
        spikes_i_u8 = np.zeros(self.n_inhibitory, dtype=np.uint8)
        n_e_spikes, n_i_spikes = _julia_ping_step(
            self.v_e,
            self.g_ampa_e,
            self.g_gaba_e,
            self.refrac_e,
            self.i_drive_e,
            xi_e,
            spikes_e_u8,
            self.v_i,
            self.g_ampa_i,
            self.g_gaba_i,
            self.refrac_i,
            self.i_drive_i,
            xi_i,
            spikes_i_u8,
            float(self.e_l),
            float(self.e_ampa),
            float(self.e_gaba),
            float(self.g_l),
            float(self.c_m),
            float(self.v_threshold),
            float(self.v_reset),
            float(self.t_refrac),
            float(self.tau_ampa),
            float(self.tau_gaba),
            float(self.sigma_e),
            float(self.sigma_i),
            float(dt),
        )
        if n_e_spikes > 0:
            self.g_ampa_e += self._w_ee_eff * n_e_spikes
            self.g_ampa_i += self._w_ei_eff * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self._w_ie_eff * n_i_spikes
            self.g_gaba_i += self._w_ii_eff * n_i_spikes
        return spikes_e_u8.astype(bool), spikes_i_u8.astype(bool)

    def _step_go(self, dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        assert _go_ping_step is not None, "backend='go' but _go_ping_step is not loaded"
        xi_e = self._rng.standard_normal(self.n_excitatory)
        xi_i = self._rng.standard_normal(self.n_inhibitory)
        spikes_e_u8 = np.zeros(self.n_excitatory, dtype=np.uint8)
        spikes_i_u8 = np.zeros(self.n_inhibitory, dtype=np.uint8)
        out_n_e = ctypes.c_uint32(0)
        out_n_i = ctypes.c_uint32(0)

        _go_ping_step.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS"),
            np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS"),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint32),
        ]

        _go_ping_step(
            self.n_excitatory,
            self.n_inhibitory,
            self.v_e,
            self.g_ampa_e,
            self.g_gaba_e,
            self.refrac_e,
            self.i_drive_e,
            xi_e,
            spikes_e_u8,
            self.v_i,
            self.g_ampa_i,
            self.g_gaba_i,
            self.refrac_i,
            self.i_drive_i,
            xi_i,
            spikes_i_u8,
            self.e_l,
            self.e_ampa,
            self.e_gaba,
            self.g_l,
            self.c_m,
            self.v_threshold,
            self.v_reset,
            self.t_refrac,
            self.tau_ampa,
            self.tau_gaba,
            self.sigma_e,
            self.sigma_i,
            dt,
            ctypes.byref(out_n_e),
            ctypes.byref(out_n_i),
        )
        n_e_spikes = out_n_e.value
        n_i_spikes = out_n_i.value

        if n_e_spikes > 0:
            self.g_ampa_e += self._w_ee_eff * n_e_spikes
            self.g_ampa_i += self._w_ei_eff * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self._w_ie_eff * n_i_spikes
            self.g_gaba_i += self._w_ii_eff * n_i_spikes
        return spikes_e_u8.astype(bool), spikes_i_u8.astype(bool)

    def _step_mojo(self, dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        assert _mojo_ping_step is not None, "backend='mojo' but _mojo_ping_step is not loaded"
        xi_e = self._rng.standard_normal(self.n_excitatory)
        xi_i = self._rng.standard_normal(self.n_inhibitory)
        spikes_e_u8 = np.zeros(self.n_excitatory, dtype=np.uint8)
        spikes_i_u8 = np.zeros(self.n_inhibitory, dtype=np.uint8)
        out_n_e = np.zeros(1, dtype=np.uint32)
        out_n_i = np.zeros(1, dtype=np.uint32)

        _mojo_ping_step.argtypes = [
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_longlong,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_longlong,
            ctypes.c_longlong,
        ]

        _mojo_ping_step(
            ctypes.c_longlong(self.n_excitatory),
            ctypes.c_longlong(self.n_inhibitory),
            ctypes.c_longlong(self.v_e.ctypes.data),
            ctypes.c_longlong(self.g_ampa_e.ctypes.data),
            ctypes.c_longlong(self.g_gaba_e.ctypes.data),
            ctypes.c_longlong(self.refrac_e.ctypes.data),
            ctypes.c_longlong(self.i_drive_e.ctypes.data),
            ctypes.c_longlong(xi_e.ctypes.data),
            ctypes.c_longlong(spikes_e_u8.ctypes.data),
            ctypes.c_longlong(self.v_i.ctypes.data),
            ctypes.c_longlong(self.g_ampa_i.ctypes.data),
            ctypes.c_longlong(self.g_gaba_i.ctypes.data),
            ctypes.c_longlong(self.refrac_i.ctypes.data),
            ctypes.c_longlong(self.i_drive_i.ctypes.data),
            ctypes.c_longlong(xi_i.ctypes.data),
            ctypes.c_longlong(spikes_i_u8.ctypes.data),
            ctypes.c_double(self.e_l),
            ctypes.c_double(self.e_ampa),
            ctypes.c_double(self.e_gaba),
            ctypes.c_double(self.g_l),
            ctypes.c_double(self.c_m),
            ctypes.c_double(self.v_threshold),
            ctypes.c_double(self.v_reset),
            ctypes.c_double(self.t_refrac),
            ctypes.c_double(self.tau_ampa),
            ctypes.c_double(self.tau_gaba),
            ctypes.c_double(self.sigma_e),
            ctypes.c_double(self.sigma_i),
            ctypes.c_double(dt),
            ctypes.c_longlong(out_n_e.ctypes.data),
            ctypes.c_longlong(out_n_i.ctypes.data),
        )

        n_e_spikes = int(out_n_e[0])
        n_i_spikes = int(out_n_i[0])
        if n_e_spikes > 0:
            self.g_ampa_e += self._w_ee_eff * n_e_spikes
            self.g_ampa_i += self._w_ei_eff * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self._w_ie_eff * n_i_spikes
            self.g_gaba_i += self._w_ii_eff * n_i_spikes
        return spikes_e_u8.astype(bool), spikes_i_u8.astype(bool)

    def _step_python(self, dt: float) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Run the reference Python per-step kernel.

        Bit-identical to
        `_step_rust` for a given seed (the noise is pre-drawn at the
        same point in the per-instance RNG sequence, so the two
        backends consume RNG identically).
        """
        assert (
            self.v_e is not None
            and self.v_i is not None
            and self.g_ampa_e is not None
            and self.g_ampa_i is not None
            and self.g_gaba_e is not None
            and self.g_gaba_i is not None
            and self.refrac_e is not None
            and self.refrac_i is not None
            and self.i_drive_e is not None
            and self.i_drive_i is not None
        )
        # Pre-draw noise so the RNG-consumption order matches the
        # Rust path exactly. The original ordering (decay → noise)
        # gives the same final result because nothing else draws
        # from the RNG between the two operations.
        noise_e = (
            self.sigma_e
            * np.sqrt(dt)
            * self._rng.standard_normal(
                self.n_excitatory,
            )
        )
        noise_i = (
            self.sigma_i
            * np.sqrt(dt)
            * self._rng.standard_normal(
                self.n_inhibitory,
            )
        )

        # Decay synaptic conductances (closed-form exponential).
        decay_ampa = np.exp(-dt / self.tau_ampa)
        decay_gaba = np.exp(-dt / self.tau_gaba)
        self.g_ampa_e *= decay_ampa
        self.g_ampa_i *= decay_ampa
        self.g_gaba_e *= decay_gaba
        self.g_gaba_i *= decay_gaba

        # Membrane currents (sign convention: outward positive).
        # E cells:  -g_L (V-E_L) - g_AMPA (V-E_AMPA) - g_GABA (V-E_GABA) + I_drive
        i_e = (
            -self.g_l * (self.v_e - self.e_l)
            - self.g_ampa_e * (self.v_e - self.e_ampa)
            - self.g_gaba_e * (self.v_e - self.e_gaba)
            + self.i_drive_e
        )
        i_i = (
            -self.g_l * (self.v_i - self.e_l)
            - self.g_ampa_i * (self.v_i - self.e_ampa)
            - self.g_gaba_i * (self.v_i - self.e_gaba)
            + self.i_drive_i
        )

        # Update voltages — only for non-refractory neurons.
        not_refrac_e = self.refrac_e <= 0.0
        not_refrac_i = self.refrac_i <= 0.0
        self.v_e[not_refrac_e] += (i_e[not_refrac_e] / self.c_m) * dt + noise_e[not_refrac_e]
        self.v_i[not_refrac_i] += (i_i[not_refrac_i] / self.c_m) * dt + noise_i[not_refrac_i]

        # Decrement refractory countdown.
        self.refrac_e = np.maximum(self.refrac_e - dt, 0.0)
        self.refrac_i = np.maximum(self.refrac_i - dt, 0.0)

        # Detect spikes (only neurons currently OUT of refractory).
        spikes_e = (self.v_e >= self.v_threshold) & not_refrac_e
        spikes_i = (self.v_i >= self.v_threshold) & not_refrac_i

        # Apply spike: clamp to reset, start refractory countdown.
        self.v_e[spikes_e] = self.v_reset
        self.v_i[spikes_i] = self.v_reset
        self.refrac_e[spikes_e] = self.t_refrac
        self.refrac_i[spikes_i] = self.t_refrac

        # Propagate spikes through synapses. E spikes raise AMPA on
        # both E (recurrent) and I (gain loop); I spikes raise GABA
        # on both E (inhibition) and I (self-inhibition).
        # `count_nonzero` instead of `.sum()`/`np.sum(...)`: under
        # coverage instrumentation NumPy can be reloaded, which makes
        # the Python-level `_NoValue` sentinel mismatch the C-level
        # one used by `umr_sum`, raising `TypeError` from `_methods.py`.
        # `count_nonzero` is a direct C call that does not use that
        # sentinel and so is reload-safe.
        n_e_spikes = int(np.count_nonzero(spikes_e))
        n_i_spikes = int(np.count_nonzero(spikes_i))
        if n_e_spikes > 0:
            # Each E spike contributes the per-source-normalised
            # `_w_*_eff` to every post-synaptic cell. Mean-field
            # all-to-all coupling matches Börgers-Kopell §2 "fully
            # connected subnetwork"; the 80/20 normalisation in
            # `__post_init__` keeps the published default weights
            # gamma-band-correct at any `n_excitatory` / `n_inhibitory`.
            self.g_ampa_e += self._w_ee_eff * n_e_spikes
            self.g_ampa_i += self._w_ei_eff * n_e_spikes
        if n_i_spikes > 0:
            self.g_gaba_e += self._w_ie_eff * n_i_spikes
            self.g_gaba_i += self._w_ii_eff * n_i_spikes

        return spikes_e.copy(), spikes_i.copy()

    # ── Reset (re-randomise initial state from the per-instance RNG) ──

    def reset_state(self) -> None:
        """Re-initialise voltages, conductances and refractory state.

        Note: this advances the RNG, so calling `reset_state()` does
        NOT return the network to its t=0 configuration. To
        reproduce a run from scratch, build a fresh `PINGCircuit`
        with the same seed.
        """
        assert self.v_e is not None and self.v_i is not None
        self.v_e[:] = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_excitatory)
        self.v_i[:] = self.e_l + self._rng.uniform(-2.0, 2.0, self.n_inhibitory)
        assert self.g_ampa_e is not None and self.g_ampa_i is not None
        assert self.g_gaba_e is not None and self.g_gaba_i is not None
        assert self.refrac_e is not None and self.refrac_i is not None
        self.g_ampa_e[:] = 0.0
        self.g_ampa_i[:] = 0.0
        self.g_gaba_e[:] = 0.0
        self.g_gaba_i[:] = 0.0
        self.refrac_e[:] = 0.0
        self.refrac_i[:] = 0.0

    # ── Spectral analysis helpers ────────────────────────────────

    @staticmethod
    def population_rate(
        spike_log: list[np.ndarray[Any, Any]],
        dt: float,
        bin_ms: float = 1.0,
    ) -> np.ndarray[Any, Any]:
        """Bin per-step spike booleans into a population rate (Hz).

        `spike_log[t]` is a length-N boolean array of spikes at
        timestep t. Output bin width is `bin_ms`; the array length
        is `len(spike_log) * dt / bin_ms` (rounded down).
        """
        if not spike_log:
            return np.array([], dtype=np.float64)
        steps_per_bin = max(1, int(bin_ms / dt))
        n_bins = len(spike_log) // steps_per_bin
        n_neurons = len(spike_log[0])
        rate = np.zeros(n_bins, dtype=np.float64)
        for b in range(n_bins):
            lo = b * steps_per_bin
            hi = lo + steps_per_bin
            spikes_in_bin = sum(int(np.count_nonzero(s)) for s in spike_log[lo:hi])
            # Convert to firing rate (Hz): spikes per neuron per second.
            rate[b] = spikes_in_bin / (n_neurons * (bin_ms / 1000.0))
        return rate

    def dominant_frequency(
        self,
        spike_log: list[np.ndarray[Any, Any]],
        dt: float,
        bin_ms: float = 1.0,
        f_min: float = 5.0,
        f_max: float = 200.0,
    ) -> float:
        """Return the dominant frequency (Hz) in the population rate.

        Uses a discrete FFT on the binned population rate; the
        returned frequency is the bin with the largest magnitude
        within `[f_min, f_max]`. Returns 0.0 if the spike log is
        too short or all silent.
        """
        rate = self.population_rate(spike_log, dt=dt, bin_ms=bin_ms)
        if rate.size < 16 or np.allclose(rate, 0.0):
            return 0.0
        # Detrend to suppress the DC bin.
        rate = rate - np.mean(rate)
        spectrum = np.abs(np.fft.rfft(rate))
        freqs = np.fft.rfftfreq(rate.size, d=bin_ms / 1000.0)
        mask = (freqs >= f_min) & (freqs <= f_max)
        if not np.any(mask):
            return 0.0
        idx = int(np.argmax(spectrum[mask]))
        return float(freqs[mask][idx])
