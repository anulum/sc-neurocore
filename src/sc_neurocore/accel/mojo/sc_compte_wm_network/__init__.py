# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Python custody facade for the native Mojo SC Compte network

"""Execute ``SC-COMPTE-WM-NETWORK`` through its complete native Mojo step.

The facade owns typed arrays, protocol currents, window aggregation, and
canonical receipts.  Mojo owns footprint construction, counter-addressed
Poisson sampling, circular recurrent aggregation, midpoint RK2 integration,
threshold/reset handling, and the atomic mutation of all 2,560 cells.  No
Python neuron recurrence is used by this execution path.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, cast

import numpy as np

from sc_neurocore.network.sc_compte_wm import (
    SCCompteWMNetworkSpec,
    summarize_activity,
)
from sc_neurocore.network.sc_compte_wm_drive import CounterPoissonReceipt
from sc_neurocore.network.sc_compte_wm_network import (
    SCCompteWMNetworkState,
    SCCompteWMRunReceipt,
    SCCompteWMStepReceipt,
    SCCompteWMStimulus,
    SCCompteWMWindowReceipt,
)

FloatArray = np.ndarray[Any, Any]
IntArray = np.ndarray[Any, Any]

LIBRARY_PATH = Path(__file__).with_name("libsc_compte_wm_network.so")
SOURCE_PATH = Path(__file__).with_name("sc_compte_wm_network.mojo")

_LIBRARY: ctypes.CDLL | None
try:
    _LIBRARY = ctypes.CDLL(str(LIBRARY_PATH))
except OSError:
    _LIBRARY = None

_HAS_MOJO_SC_COMPTE_WM_NETWORK = _LIBRARY is not None
_SPECTRUM_CACHE: dict[tuple[float, float], tuple[FloatArray, FloatArray]] = {}

if _LIBRARY is not None:
    _LIBRARY.sc_compte_wm_network_kernel_spectrum_c.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_int64,
    ]
    _LIBRARY.sc_compte_wm_network_kernel_spectrum_c.restype = ctypes.c_int32
    _LIBRARY.sc_compte_wm_network_counter_poisson_c.argtypes = [
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_uint64,
        ctypes.c_int64,
    ]
    _LIBRARY.sc_compte_wm_network_counter_poisson_c.restype = ctypes.c_int32
    _LIBRARY.sc_compte_wm_network_step_c.argtypes = [
        *([ctypes.c_int64] * 15),
        ctypes.c_double,
        *([ctypes.c_int64] * 6),
    ]
    _LIBRARY.sc_compte_wm_network_step_c.restype = ctypes.c_int32


def _array_digest(hasher: Any, values: np.ndarray[Any, Any], dtype: str) -> None:
    hasher.update(np.ascontiguousarray(values, dtype=dtype).tobytes())


def _address(values: np.ndarray[Any, Any]) -> int:
    return int(values.ctypes.data)


@dataclass(frozen=True, slots=True)
class MojoRuntimeProvenance:
    """Hashes binding one loaded Mojo source and shared library."""

    source_sha256: str
    library_sha256: str


class SCCompteWMMojoNetwork:
    """Run the fixed SC 2,560-cell working-memory network through Mojo.

    Only ``seed``, ``structured_ei``, ``modulated``, and
    ``allow_recurrent_autapses`` are runtime-selectable in the v1 native ABI.
    Other specification changes fail closed because the native kernel encodes
    the frozen v1 constants directly.
    """

    _V_MIN = -200.0
    _V_MAX = 100.0
    _GATE_MAX = 1.0e6
    _UINT64_MAX = (1 << 64) - 1
    _NATIVE_FIXED_FIELDS = (
        "n_excitatory",
        "n_inhibitory",
        "dt_ms",
        "external_rate_hz",
        "external_exc_conductance_ns",
        "external_inh_conductance_ns",
        "recurrent_ee_conductance_ns",
        "recurrent_ei_conductance_ns",
        "recurrent_ie_conductance_ns",
        "recurrent_ii_conductance_ns",
        "ee_j_plus",
        "ee_sigma_deg",
        "ei_j_plus",
        "ei_sigma_deg",
        "tau_ampa_ms",
        "tau_nmda_ms",
        "tau_nmda_rise_ms",
        "alpha_nmda_per_ms",
        "tau_gabaa_ms",
        "magnesium_mm",
        "excitatory",
        "inhibitory",
    )

    def __init__(
        self,
        spec: SCCompteWMNetworkSpec | None = None,
        *,
        state: SCCompteWMNetworkState | None = None,
    ) -> None:
        if _LIBRARY is None:
            raise RuntimeError("the repository-local Mojo SC Compte shared library is unavailable")
        self.spec = SCCompteWMNetworkSpec() if spec is None else spec
        baseline = SCCompteWMNetworkSpec()
        changed = [
            name
            for name in self._NATIVE_FIXED_FIELDS
            if getattr(self.spec, name) != getattr(baseline, name)
        ]
        if changed:
            raise ValueError(
                "the Mojo v1 ABI fixes these specification fields: " + ", ".join(changed)
            )
        self._ee_spectrum_real, self._ee_spectrum_imag = self._spectrum(
            self.spec.ee_j_plus, self.spec.ee_sigma_deg
        )
        self._ei_spectrum_real, self._ei_spectrum_imag = self._spectrum(
            self.spec.ei_j_plus, self.spec.ei_sigma_deg
        )
        self._ee_kernel_zero = float(
            self.spec.connectivity_footprint(
                "ee", 0.0, self.spec.preferred_angles_deg("excitatory")
            )[0]
        )
        self._state = self._initial_state() if state is None else state.copy()
        self._validate_state(self._state)

    @property
    def provenance(self) -> MojoRuntimeProvenance:
        """Return exact source and loaded-library digests."""
        return MojoRuntimeProvenance(
            source_sha256=hashlib.sha256(SOURCE_PATH.read_bytes()).hexdigest(),
            library_sha256=hashlib.sha256(LIBRARY_PATH.read_bytes()).hexdigest(),
        )

    def _spectrum(self, j_plus: float, sigma_deg: float) -> tuple[FloatArray, FloatArray]:
        cached = _SPECTRUM_CACHE.get((j_plus, sigma_deg))
        if cached is not None:
            return cached
        real = np.empty(self.spec.n_excitatory, dtype=np.float64)
        imag = np.empty(self.spec.n_excitatory, dtype=np.float64)
        assert _LIBRARY is not None
        status = int(
            _LIBRARY.sc_compte_wm_network_kernel_spectrum_c(
                j_plus, sigma_deg, _address(real), _address(imag)
            )
        )
        if status != 0:
            raise RuntimeError(f"Mojo footprint construction failed with status {status}")
        real.flags.writeable = False
        imag.flags.writeable = False
        _SPECTRUM_CACHE[(j_plus, sigma_deg)] = (real, imag)
        return real, imag

    def _initial_state(self) -> SCCompteWMNetworkState:
        exc = self.spec.n_excitatory
        inh = self.spec.n_inhibitory
        return SCCompteWMNetworkState(
            step_index=0,
            v_exc_mv=np.full(exc, -70.0, dtype=np.float64),
            v_inh_mv=np.full(inh, -70.0, dtype=np.float64),
            refractory_exc_ms=np.zeros(exc, dtype=np.float64),
            refractory_inh_ms=np.zeros(inh, dtype=np.float64),
            external_ampa_exc=np.zeros(exc, dtype=np.float64),
            external_ampa_inh=np.zeros(inh, dtype=np.float64),
            recurrent_nmda=np.zeros(exc, dtype=np.float64),
            recurrent_nmda_rise=np.zeros(exc, dtype=np.float64),
            recurrent_gabaa=np.zeros(inh, dtype=np.float64),
        )

    def state(self) -> SCCompteWMNetworkState:
        """Return a defensive deep copy of every network state array."""
        return self._state.copy()

    def reset(self) -> None:
        """Restore the frozen initial state while retaining the specification."""
        self._state = self._initial_state()

    def _validate_state(self, state: SCCompteWMNetworkState) -> None:
        if (
            isinstance(state.step_index, bool)
            or not isinstance(state.step_index, int)
            or not 0 <= state.step_index < self._UINT64_MAX
        ):
            raise ValueError("state step_index is outside the executable uint64 range")
        shapes = {
            "v_exc_mv": (2048,),
            "v_inh_mv": (512,),
            "refractory_exc_ms": (2048,),
            "refractory_inh_ms": (512,),
            "external_ampa_exc": (2048,),
            "external_ampa_inh": (512,),
            "recurrent_nmda": (2048,),
            "recurrent_nmda_rise": (2048,),
            "recurrent_gabaa": (512,),
        }
        for name, shape in shapes.items():
            values = np.asarray(getattr(state, name))
            if (
                values.shape != shape
                or values.dtype != np.float64
                or not values.flags.c_contiguous
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(
                    f"{name} must be a finite contiguous float64 array with shape {shape}"
                )
        if np.any(state.v_exc_mv < self._V_MIN) or np.any(state.v_exc_mv > self._V_MAX):
            raise ValueError("excitatory voltage lies outside the safety envelope")
        if np.any(state.v_inh_mv < self._V_MIN) or np.any(state.v_inh_mv > self._V_MAX):
            raise ValueError("inhibitory voltage lies outside the safety envelope")
        gates = (
            state.refractory_exc_ms,
            state.refractory_inh_ms,
            state.external_ampa_exc,
            state.external_ampa_inh,
            state.recurrent_nmda,
            state.recurrent_nmda_rise,
            state.recurrent_gabaa,
        )
        if any(np.any(values < 0.0) or np.any(values > self._GATE_MAX) for values in gates):
            raise ValueError("refractory/channel state lies outside the safety envelope")
        if np.any(state.recurrent_nmda > 1.0):
            raise ValueError("recurrent NMDA gates must remain bounded by one")

    @staticmethod
    def _explicit_receipt(events: IntArray, step_index: int, stream: int) -> CounterPoissonReceipt:
        canonical = np.ascontiguousarray(events, dtype="<i8")
        return CounterPoissonReceipt(
            step_index=step_index,
            stream=stream,
            population_size=len(events),
            total_events=int(np.sum(events, dtype=np.int64)),
            event_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
        )

    def _events(self, name: str, values: Any, size: int) -> IntArray:
        array = np.asarray(values)
        if array.shape != (size,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be an integer array with shape ({size},)")
        if np.any(array < 0) or np.any(array > self._GATE_MAX):
            raise ValueError(f"{name} must contain bounded non-negative event counts")
        return cast(IntArray, np.ascontiguousarray(array, dtype=np.uint64))

    def _counter_events(
        self, population_size: int, step_index: int, stream: int
    ) -> tuple[IntArray, CounterPoissonReceipt]:
        events = np.empty(population_size, dtype=np.uint64)
        assert _LIBRARY is not None
        status = int(
            _LIBRARY.sc_compte_wm_network_counter_poisson_c(
                population_size,
                self.spec.external_rate_hz,
                self.spec.dt_ms,
                self.spec.seed,
                stream,
                step_index,
                _address(events),
            )
        )
        if status != 0:
            raise RuntimeError(f"Mojo counter-Poisson sampling failed with status {status}")
        receipt = self._explicit_receipt(events, step_index, stream)
        return events, receipt

    def _current(self, values: Any | None) -> FloatArray:
        if values is None:
            return np.zeros(2048, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (2048,) or not np.all(np.isfinite(array)):
            raise ValueError("direct_exc_current_pa must be finite with shape (2048,)")
        return cast(FloatArray, np.ascontiguousarray(array))

    def step(
        self,
        direct_exc_current_pa: Any | None = None,
        *,
        external_exc_events: Any | None = None,
        external_inh_events: Any | None = None,
    ) -> SCCompteWMStepReceipt:
        """Advance one native atomic step and return portable receipts."""
        self._validate_state(self._state)
        if (external_exc_events is None) != (external_inh_events is None):
            raise ValueError("explicit external event arrays must be supplied together")
        current = self._current(direct_exc_current_pa)
        step_index = self._state.step_index
        if external_exc_events is None:
            exc_events, exc_receipt = self._counter_events(2048, step_index, 0)
            inh_events, inh_receipt = self._counter_events(512, step_index, 1)
        else:
            exc_events = self._events("external_exc_events", external_exc_events, 2048)
            inh_events = self._events("external_inh_events", external_inh_events, 512)
            exc_receipt = self._explicit_receipt(exc_events, step_index, 0)
            inh_receipt = self._explicit_receipt(inh_events, step_index, 1)
        exc_spikes_i64 = np.empty(2048, dtype=np.int64)
        inh_spikes_i64 = np.empty(512, dtype=np.int64)
        state = self._state
        assert _LIBRARY is not None
        status = int(
            _LIBRARY.sc_compte_wm_network_step_c(
                int(self.spec.structured_ei),
                int(self.spec.modulated),
                int(self.spec.allow_recurrent_autapses),
                _address(state.v_exc_mv),
                _address(state.v_inh_mv),
                _address(state.refractory_exc_ms),
                _address(state.refractory_inh_ms),
                _address(state.external_ampa_exc),
                _address(state.external_ampa_inh),
                _address(state.recurrent_nmda),
                _address(state.recurrent_nmda_rise),
                _address(state.recurrent_gabaa),
                _address(current),
                _address(exc_events),
                _address(inh_events),
                self._ee_kernel_zero,
                _address(self._ee_spectrum_real),
                _address(self._ee_spectrum_imag),
                _address(self._ei_spectrum_real),
                _address(self._ei_spectrum_imag),
                _address(exc_spikes_i64),
                _address(inh_spikes_i64),
            )
        )
        if status != 0:
            raise ValueError(f"Mojo atomic network step rejected candidate with status {status}")
        state.step_index += 1
        exc_spikes = exc_spikes_i64.astype(np.bool_)
        inh_spikes = inh_spikes_i64.astype(np.bool_)
        input_digest = hashlib.sha256()
        _array_digest(input_digest, exc_events, "<i8")
        _array_digest(input_digest, inh_events, "<i8")
        _array_digest(input_digest, current, "<f8")
        return SCCompteWMStepReceipt(
            step_index=step_index,
            excitatory_spikes=exc_spikes,
            inhibitory_spikes=inh_spikes,
            excitatory_input=exc_receipt,
            inhibitory_input=inh_receipt,
            input_sha256=input_digest.hexdigest(),
            state_sha256=state.sha256(),
        )

    def _stimulus_current(
        self, time_ms: float, stimuli: tuple[SCCompteWMStimulus, ...]
    ) -> FloatArray:
        current = np.zeros(2048, dtype=np.float64)
        angles = self.spec.preferred_angles_deg("excitatory")
        for stimulus in stimuli:
            if stimulus.start_ms <= time_ms < stimulus.start_ms + stimulus.duration_ms:
                if stimulus.kind == "global_current":
                    current += stimulus.current_pa
                else:
                    assert stimulus.center_deg is not None
                    current += self.spec.cue_current_pa(
                        stimulus.center_deg, angles, peak_pa=stimulus.current_pa
                    )
        return current

    def run(
        self,
        duration_ms: float,
        *,
        stimuli: tuple[SCCompteWMStimulus, ...] = (),
        statistics_window_ms: float | None = None,
    ) -> SCCompteWMRunReceipt:
        """Execute a bounded native run and aggregate window statistics."""
        if not math.isfinite(duration_ms) or duration_ms <= 0.0:
            raise ValueError("duration_ms must be finite and positive")
        raw_steps = duration_ms / self.spec.dt_ms
        steps = round(raw_steps)
        if not math.isclose(raw_steps, steps, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("duration_ms must be an integral number of network timesteps")
        window_ms = (
            self.spec.protocol.statistics_window_ms
            if statistics_window_ms is None
            else statistics_window_ms
        )
        if not math.isfinite(window_ms) or window_ms <= 0.0:
            raise ValueError("statistics_window_ms must be finite and positive")
        raw_window_steps = window_ms / self.spec.dt_ms
        window_steps = round(raw_window_steps)
        if not math.isclose(raw_window_steps, window_steps, rel_tol=0.0, abs_tol=1.0e-10):
            raise ValueError("statistics_window_ms must be an integral number of timesteps")
        for stimulus in stimuli:
            if stimulus.start_ms + stimulus.duration_ms > duration_ms + 1.0e-12:
                raise ValueError("stimulus epochs must lie within the requested run")

        input_digest = hashlib.sha256()
        spike_digest = hashlib.sha256()
        exc_window = np.zeros(2048, dtype=np.int64)
        inh_window = np.zeros(512, dtype=np.int64)
        total_exc = 0
        total_inh = 0
        windows: list[SCCompteWMWindowReceipt] = []
        window_start_step = 0
        for offset in range(steps):
            receipt = self.step(self._stimulus_current(offset * self.spec.dt_ms, stimuli))
            input_digest.update(bytes.fromhex(receipt.input_sha256))
            _array_digest(spike_digest, receipt.excitatory_spikes, "|b1")
            _array_digest(spike_digest, receipt.inhibitory_spikes, "|b1")
            exc_window += receipt.excitatory_spikes
            inh_window += receipt.inhibitory_spikes
            total_exc += int(np.sum(receipt.excitatory_spikes))
            total_inh += int(np.sum(receipt.inhibitory_spikes))
            boundary = (offset + 1) % window_steps == 0 or offset + 1 == steps
            if boundary:
                elapsed_steps = offset + 1 - window_start_step
                elapsed_ms = elapsed_steps * self.spec.dt_ms
                statistics = (
                    summarize_activity(self.spec, exc_window, inh_window, elapsed_ms)
                    if int(np.sum(exc_window)) > 0
                    else None
                )
                windows.append(
                    SCCompteWMWindowReceipt(
                        start_ms=window_start_step * self.spec.dt_ms,
                        end_ms=(offset + 1) * self.spec.dt_ms,
                        excitatory_spikes=int(np.sum(exc_window)),
                        inhibitory_spikes=int(np.sum(inh_window)),
                        statistics=statistics,
                    )
                )
                exc_window.fill(0)
                inh_window.fill(0)
                window_start_step = offset + 1
        return SCCompteWMRunReceipt(
            specification_version=self.spec.specification_version,
            seed=self.spec.seed,
            duration_ms=duration_ms,
            steps=steps,
            excitatory_spikes=total_exc,
            inhibitory_spikes=total_inh,
            windows=tuple(windows),
            input_sha256=input_digest.hexdigest(),
            spike_sha256=spike_digest.hexdigest(),
            final_state_sha256=self._state.sha256(),
        )


__all__ = [
    "LIBRARY_PATH",
    "MojoRuntimeProvenance",
    "SCCompteWMMojoNetwork",
    "SOURCE_PATH",
    "_HAS_MOJO_SC_COMPTE_WM_NETWORK",
]
