# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — executable 2,560-cell Compte working-memory network

"""Deterministic Python execution of the SC Compte working-memory network.

This is the separately named SC network modification, not the preserved
single-cell ``CompteWMNeuron``.  It enrols all 2,048 excitatory and 512
inhibitory cells, source-unit conductances, external Poisson drive, circular
connectivity, midpoint RK2 channel flow, sampled threshold/reset behaviour,
protocol currents, and bounded receipts.  A run is executable evidence only;
persistent-bump and distractor-resistance claims still require the planned
multi-seed ensemble acceptance layer.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Literal, cast

import numpy as np

from .sc_compte_wm import (
    SCCompteWMActivityStatistics,
    SCCompteWMNetworkSpec,
    summarize_activity,
)
from .sc_compte_wm_drive import CounterPoissonDrive, CounterPoissonReceipt

FloatArray = np.ndarray[Any, Any]
IntArray = np.ndarray[Any, Any]
BoolArray = np.ndarray[Any, Any]
StimulusKind = Literal["localized_cue", "global_current"]


def _array_digest(hasher: Any, values: np.ndarray[Any, Any], dtype: str) -> None:
    hasher.update(np.ascontiguousarray(values, dtype=dtype).tobytes())


def _current_digest(hasher: Any, current_pa: np.ndarray[Any, Any]) -> None:
    """Hash current at 1e-9 pA resolution, independent of libm rounding."""
    quantized = np.floor(current_pa * 1_000_000_000.0 + 0.5)
    _array_digest(hasher, quantized, "<i8")


@dataclass(slots=True)
class SCCompteWMNetworkState:
    """Complete mutable state of the 2,560-cell network."""

    step_index: int
    v_exc_mv: FloatArray
    v_inh_mv: FloatArray
    refractory_exc_ms: FloatArray
    refractory_inh_ms: FloatArray
    external_ampa_exc: FloatArray
    external_ampa_inh: FloatArray
    recurrent_nmda: FloatArray
    recurrent_nmda_rise: FloatArray
    recurrent_gabaa: FloatArray

    def copy(self) -> SCCompteWMNetworkState:
        """Return a deep state copy suitable for checkpointing."""
        return SCCompteWMNetworkState(
            step_index=self.step_index,
            v_exc_mv=self.v_exc_mv.copy(),
            v_inh_mv=self.v_inh_mv.copy(),
            refractory_exc_ms=self.refractory_exc_ms.copy(),
            refractory_inh_ms=self.refractory_inh_ms.copy(),
            external_ampa_exc=self.external_ampa_exc.copy(),
            external_ampa_inh=self.external_ampa_inh.copy(),
            recurrent_nmda=self.recurrent_nmda.copy(),
            recurrent_nmda_rise=self.recurrent_nmda_rise.copy(),
            recurrent_gabaa=self.recurrent_gabaa.copy(),
        )

    def sha256(self) -> str:
        """Return a canonical digest of every state scalar and array."""
        digest = hashlib.sha256()
        digest.update(int(self.step_index).to_bytes(8, "little", signed=False))
        for values in (
            self.v_exc_mv,
            self.v_inh_mv,
            self.refractory_exc_ms,
            self.refractory_inh_ms,
            self.external_ampa_exc,
            self.external_ampa_inh,
            self.recurrent_nmda,
            self.recurrent_nmda_rise,
            self.recurrent_gabaa,
        ):
            _array_digest(digest, values, "<f8")
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class SCCompteWMStimulus:
    """One bounded excitatory-population current epoch in source pA units."""

    start_ms: float
    duration_ms: float
    current_pa: float
    kind: StimulusKind = "localized_cue"
    center_deg: float | None = 0.0

    def __post_init__(self) -> None:
        for name in ("start_ms", "duration_ms", "current_pa"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or (name != "start_ms" and value <= 0.0) or value < 0.0:
                raise ValueError(
                    f"{name} must be finite and non-negative, with positive duration/current"
                )
        if self.kind not in ("localized_cue", "global_current"):
            raise ValueError("kind must be 'localized_cue' or 'global_current'")
        if self.kind == "localized_cue":
            if self.center_deg is None or not math.isfinite(self.center_deg):
                raise ValueError("localized cues require a finite center_deg")
        elif self.center_deg is not None:
            raise ValueError("global currents require center_deg=None")


@dataclass(frozen=True, slots=True)
class SCCompteWMStepReceipt:
    """Events and provenance emitted by one atomic network step."""

    step_index: int
    excitatory_spikes: BoolArray
    inhibitory_spikes: BoolArray
    excitatory_input: CounterPoissonReceipt
    inhibitory_input: CounterPoissonReceipt
    input_sha256: str
    state_sha256: str


@dataclass(frozen=True, slots=True)
class SCCompteWMWindowReceipt:
    """Population statistics for one explicit run window."""

    start_ms: float
    end_ms: float
    excitatory_spikes: int
    inhibitory_spikes: int
    statistics: SCCompteWMActivityStatistics | None


@dataclass(frozen=True, slots=True)
class SCCompteWMRunReceipt:
    """Bounded aggregate evidence from one network execution."""

    specification_version: str
    seed: int
    duration_ms: float
    steps: int
    excitatory_spikes: int
    inhibitory_spikes: int
    windows: tuple[SCCompteWMWindowReceipt, ...]
    input_sha256: str
    spike_sha256: str
    final_state_sha256: str


class SCCompteWMNetwork:
    """Execute the frozen SC 2,560-cell network with deterministic receipts."""

    _V_MIN = -200.0
    _V_MAX = 100.0
    _GATE_MAX = 1.0e6

    def __init__(
        self,
        spec: SCCompteWMNetworkSpec | None = None,
        *,
        state: SCCompteWMNetworkState | None = None,
    ) -> None:
        self.spec = SCCompteWMNetworkSpec() if spec is None else spec
        self._drive_exc = CounterPoissonDrive(
            self.spec.n_excitatory,
            self.spec.external_rate_hz,
            self.spec.dt_ms,
            self.spec.seed,
            0,
        )
        self._drive_inh = CounterPoissonDrive(
            self.spec.n_inhibitory,
            self.spec.external_rate_hz,
            self.spec.dt_ms,
            self.spec.seed,
            1,
        )
        angles = self.spec.preferred_angles_deg("excitatory")
        self._ee_kernel = self.spec.connectivity_footprint("ee", 0.0, angles)
        self._ee_fft = np.fft.rfft(self._ee_kernel)
        self._ei_fft = (
            np.fft.rfft(self.spec.connectivity_footprint("ei", 0.0, angles))
            if self.spec.structured_ei
            else None
        )
        self._state = self._initial_state() if state is None else state.copy()
        self._validate_state(self._state)

    def _initial_state(self) -> SCCompteWMNetworkState:
        exc = self.spec.n_excitatory
        inh = self.spec.n_inhibitory

        def zeros_exc() -> FloatArray:
            return np.zeros(exc, dtype=np.float64)

        def zeros_inh() -> FloatArray:
            return np.zeros(inh, dtype=np.float64)

        return SCCompteWMNetworkState(
            step_index=0,
            v_exc_mv=np.full(exc, self.spec.excitatory.leak_reversal_mv, dtype=np.float64),
            v_inh_mv=np.full(inh, self.spec.inhibitory.leak_reversal_mv, dtype=np.float64),
            refractory_exc_ms=zeros_exc(),
            refractory_inh_ms=zeros_inh(),
            external_ampa_exc=zeros_exc(),
            external_ampa_inh=zeros_inh(),
            recurrent_nmda=zeros_exc(),
            recurrent_nmda_rise=zeros_exc(),
            recurrent_gabaa=zeros_inh(),
        )

    def state(self) -> SCCompteWMNetworkState:
        """Return a deep copy of the complete current state."""
        return self._state.copy()

    def reset(self) -> None:
        """Reset dynamic state while preserving the specification and streams."""
        self._state = self._initial_state()

    def _validate_state(self, state: SCCompteWMNetworkState) -> None:
        if isinstance(state.step_index, bool) or not 0 <= state.step_index <= (1 << 64) - 2:
            raise ValueError("state step_index is outside the executable uint64 range")
        shapes = {
            "v_exc_mv": (self.spec.n_excitatory,),
            "v_inh_mv": (self.spec.n_inhibitory,),
            "refractory_exc_ms": (self.spec.n_excitatory,),
            "refractory_inh_ms": (self.spec.n_inhibitory,),
            "external_ampa_exc": (self.spec.n_excitatory,),
            "external_ampa_inh": (self.spec.n_inhibitory,),
            "recurrent_nmda": (self.spec.n_excitatory,),
            "recurrent_nmda_rise": (self.spec.n_excitatory,),
            "recurrent_gabaa": (self.spec.n_inhibitory,),
        }
        for name, shape in shapes.items():
            values = np.asarray(getattr(state, name))
            if (
                values.shape != shape
                or values.dtype != np.float64
                or not np.all(np.isfinite(values))
            ):
                raise ValueError(f"{name} must be a finite float64 array with shape {shape}")
        if np.any(state.v_exc_mv < self._V_MIN) or np.any(state.v_exc_mv > self._V_MAX):
            raise ValueError("excitatory voltage lies outside the safety envelope")
        if np.any(state.v_inh_mv < self._V_MIN) or np.any(state.v_inh_mv > self._V_MAX):
            raise ValueError("inhibitory voltage lies outside the safety envelope")
        for values in (
            state.refractory_exc_ms,
            state.refractory_inh_ms,
            state.external_ampa_exc,
            state.external_ampa_inh,
            state.recurrent_nmda,
            state.recurrent_nmda_rise,
            state.recurrent_gabaa,
        ):
            if np.any(values < 0.0) or np.any(values > self._GATE_MAX):
                raise ValueError("refractory/channel state lies outside the safety envelope")
        if np.any(state.recurrent_nmda > 1.0):
            raise ValueError("recurrent NMDA gates must remain bounded by one")

    @staticmethod
    def _mg_block(voltage_mv: FloatArray, magnesium_mm: float) -> FloatArray:
        exponent = np.clip(-0.062 * voltage_mv, -700.0, 700.0)
        return cast(FloatArray, 1.0 / (1.0 + magnesium_mm / 3.57 * np.exp(exponent)))

    def _circular_sum(self, source: FloatArray, kernel_fft: FloatArray) -> FloatArray:
        values = np.fft.irfft(np.fft.rfft(source) * kernel_fft, n=self.spec.n_excitatory)
        return cast(FloatArray, values)

    def _recurrent_aggregates(
        self, nmda: FloatArray, gabaa: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        ee = self._circular_sum(nmda, self._ee_fft)
        if not self.spec.allow_recurrent_autapses:
            ee = ee - self._ee_kernel[0] * nmda
        if self._ei_fft is None:
            ei = np.full(self.spec.n_inhibitory, float(np.sum(nmda)), dtype=np.float64)
        else:
            ei = cast(FloatArray, self._circular_sum(nmda, self._ei_fft)[::4].copy())
        total_gabaa = float(np.sum(gabaa))
        ie = np.full(self.spec.n_excitatory, total_gabaa, dtype=np.float64)
        ii = np.full(self.spec.n_inhibitory, total_gabaa, dtype=np.float64)
        if not self.spec.allow_recurrent_autapses:
            ii -= gabaa
        return ee, ei, ie, ii

    def _derivatives(
        self,
        v_exc: FloatArray,
        v_inh: FloatArray,
        ext_exc: FloatArray,
        ext_inh: FloatArray,
        nmda: FloatArray,
        nmda_rise: FloatArray,
        gabaa: FloatArray,
        current_exc_na: FloatArray,
        active_exc: BoolArray,
        active_inh: BoolArray,
    ) -> tuple[FloatArray, ...]:
        ee, ei, ie, ii = self._recurrent_aggregates(nmda, gabaa)
        g_ext_exc = self.spec.external_exc_conductance_ns / 1000.0
        g_ext_inh = self.spec.external_inh_conductance_ns / 1000.0
        g_ee = self.spec.recurrent_conductance_ns("ee") / 1000.0
        g_ei = self.spec.recurrent_conductance_ns("ei") / 1000.0
        g_ie = self.spec.recurrent_conductance_ns("ie") / 1000.0
        g_ii = self.spec.recurrent_conductance_ns("ii") / 1000.0
        leak_exc = self.spec.excitatory.leak_conductance_ns / 1000.0
        leak_inh = self.spec.inhibitory.leak_conductance_ns / 1000.0
        dv_exc = (
            -leak_exc * (v_exc - self.spec.excitatory.leak_reversal_mv)
            - g_ext_exc * ext_exc * v_exc
            - g_ee * ee * self._mg_block(v_exc, self.spec.magnesium_mm) * v_exc
            - g_ie * ie * (v_exc + 70.0)
            + current_exc_na
        ) / self.spec.excitatory.capacitance_nf
        dv_inh = (
            -leak_inh * (v_inh - self.spec.inhibitory.leak_reversal_mv)
            - g_ext_inh * ext_inh * v_inh
            - g_ei * ei * self._mg_block(v_inh, self.spec.magnesium_mm) * v_inh
            - g_ii * ii * (v_inh + 70.0)
        ) / self.spec.inhibitory.capacitance_nf
        dv_exc = np.where(active_exc, dv_exc, 0.0)
        dv_inh = np.where(active_inh, dv_inh, 0.0)
        return (
            cast(FloatArray, dv_exc),
            cast(FloatArray, dv_inh),
            cast(FloatArray, -ext_exc / self.spec.tau_ampa_ms),
            cast(FloatArray, -ext_inh / self.spec.tau_ampa_ms),
            cast(
                FloatArray,
                -nmda / self.spec.tau_nmda_ms
                + self.spec.alpha_nmda_per_ms * nmda_rise * (1.0 - nmda),
            ),
            cast(FloatArray, -nmda_rise / self.spec.tau_nmda_rise_ms),
            cast(FloatArray, -gabaa / self.spec.tau_gabaa_ms),
        )

    def _events(self, name: str, values: Any, size: int) -> IntArray:
        array = np.asarray(values)
        if array.shape != (size,) or not np.issubdtype(array.dtype, np.integer):
            raise ValueError(f"{name} must be an integer array with shape ({size},)")
        if np.any(array < 0):
            raise ValueError(f"{name} must contain non-negative event counts")
        return cast(IntArray, array.astype(np.int64, copy=True))

    def _current(self, values: Any | None) -> FloatArray:
        if values is None:
            return np.zeros(self.spec.n_excitatory, dtype=np.float64)
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (self.spec.n_excitatory,) or not np.all(np.isfinite(array)):
            raise ValueError("direct_exc_current_pa must be finite with shape (2048,)")
        return cast(FloatArray, array.copy())

    def step(
        self,
        direct_exc_current_pa: Any | None = None,
        *,
        external_exc_events: Any | None = None,
        external_inh_events: Any | None = None,
    ) -> SCCompteWMStepReceipt:
        """Advance one atomic midpoint-RK2 step and return complete event receipts.

        Explicit external event arrays bypass the counter drive and provide an
        independent-oracle/co-simulation boundary.  Both arrays must be
        supplied together.  All validation and candidate checks precede the
        state mutation.
        """
        self._validate_state(self._state)
        current_pa = self._current(direct_exc_current_pa)
        if (external_exc_events is None) != (external_inh_events is None):
            raise ValueError("explicit external event arrays must be supplied together")
        step_index = self._state.step_index
        if external_exc_events is None:
            exc_events, exc_receipt = self._drive_exc.sample(step_index)
            inh_events, inh_receipt = self._drive_inh.sample(step_index)
        else:
            exc_events = self._events("external_exc_events", external_exc_events, 2048)
            inh_events = self._events("external_inh_events", external_inh_events, 512)
            exc_receipt = self._explicit_receipt(exc_events, step_index, 0)
            inh_receipt = self._explicit_receipt(inh_events, step_index, 1)

        state = self._state
        start = (
            state.v_exc_mv,
            state.v_inh_mv,
            state.external_ampa_exc + exc_events,
            state.external_ampa_inh + inh_events,
            state.recurrent_nmda,
            state.recurrent_nmda_rise,
            state.recurrent_gabaa,
        )
        active_exc = state.refractory_exc_ms <= 0.0
        active_inh = state.refractory_inh_ms <= 0.0
        current_na = current_pa / 1000.0
        k1 = self._derivatives(*start, current_na, active_exc, active_inh)
        midpoint = cast(
            tuple[
                FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray
            ],
            tuple(
                value + 0.5 * self.spec.dt_ms * slope
                for value, slope in zip(start, k1, strict=True)
            ),
        )
        k2 = self._derivatives(*midpoint, current_na, active_exc, active_inh)
        candidate = tuple(
            value + self.spec.dt_ms * slope for value, slope in zip(start, k2, strict=True)
        )
        v_exc, v_inh, ext_exc, ext_inh, nmda, nmda_rise, gabaa = candidate
        ref_exc = np.maximum(0.0, state.refractory_exc_ms - self.spec.dt_ms)
        ref_inh = np.maximum(0.0, state.refractory_inh_ms - self.spec.dt_ms)
        v_exc = np.where(active_exc, v_exc, self.spec.excitatory.reset_mv)
        v_inh = np.where(active_inh, v_inh, self.spec.inhibitory.reset_mv)
        exc_spikes = cast(BoolArray, active_exc & (v_exc >= self.spec.excitatory.threshold_mv))
        inh_spikes = cast(BoolArray, active_inh & (v_inh >= self.spec.inhibitory.threshold_mv))
        v_exc = np.where(exc_spikes, self.spec.excitatory.reset_mv, v_exc)
        v_inh = np.where(inh_spikes, self.spec.inhibitory.reset_mv, v_inh)
        ref_exc = np.where(exc_spikes, self.spec.excitatory.refractory_ms, ref_exc)
        ref_inh = np.where(inh_spikes, self.spec.inhibitory.refractory_ms, ref_inh)
        nmda_rise = nmda_rise + exc_spikes
        gabaa = gabaa + inh_spikes
        next_state = SCCompteWMNetworkState(
            step_index=step_index + 1,
            v_exc_mv=cast(FloatArray, np.asarray(v_exc, dtype=np.float64)),
            v_inh_mv=cast(FloatArray, np.asarray(v_inh, dtype=np.float64)),
            refractory_exc_ms=cast(FloatArray, np.asarray(ref_exc, dtype=np.float64)),
            refractory_inh_ms=cast(FloatArray, np.asarray(ref_inh, dtype=np.float64)),
            external_ampa_exc=cast(FloatArray, np.asarray(ext_exc, dtype=np.float64)),
            external_ampa_inh=cast(FloatArray, np.asarray(ext_inh, dtype=np.float64)),
            recurrent_nmda=cast(FloatArray, np.asarray(nmda, dtype=np.float64)),
            recurrent_nmda_rise=cast(FloatArray, np.asarray(nmda_rise, dtype=np.float64)),
            recurrent_gabaa=cast(FloatArray, np.asarray(gabaa, dtype=np.float64)),
        )
        self._validate_state(next_state)
        input_digest = hashlib.sha256()
        _array_digest(input_digest, exc_events, "<i8")
        _array_digest(input_digest, inh_events, "<i8")
        _current_digest(input_digest, current_pa)
        self._state = next_state
        return SCCompteWMStepReceipt(
            step_index=step_index,
            excitatory_spikes=exc_spikes.copy(),
            inhibitory_spikes=inh_spikes.copy(),
            excitatory_input=exc_receipt,
            inhibitory_input=inh_receipt,
            input_sha256=input_digest.hexdigest(),
            state_sha256=next_state.sha256(),
        )

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

    def _stimulus_current(
        self, time_ms: float, stimuli: tuple[SCCompteWMStimulus, ...]
    ) -> FloatArray:
        current = np.zeros(self.spec.n_excitatory, dtype=np.float64)
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
        """Execute an integral number of steps and return bounded run evidence."""
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
        exc_window = np.zeros(self.spec.n_excitatory, dtype=np.int64)
        inh_window = np.zeros(self.spec.n_inhibitory, dtype=np.int64)
        total_exc = 0
        total_inh = 0
        windows: list[SCCompteWMWindowReceipt] = []
        window_start_step = 0
        for offset in range(steps):
            current = self._stimulus_current(offset * self.spec.dt_ms, stimuli)
            receipt = self.step(current)
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
    "SCCompteWMNetwork",
    "SCCompteWMNetworkState",
    "SCCompteWMRunReceipt",
    "SCCompteWMStepReceipt",
    "SCCompteWMStimulus",
    "SCCompteWMWindowReceipt",
]
