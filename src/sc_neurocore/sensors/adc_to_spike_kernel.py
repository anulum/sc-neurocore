# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike window rate-code — bit-true reference + dispatch

"""Bit-true integer reference for the ADC-to-spike decimating rate-code encoder.

Each decimation window of raw ADC samples is centred and quantised to a Q-format
code, sign-aware averaged, and converted into a deterministic rate code: the
spike count is ``|window| // threshold`` and the polarity is the window sign. This
is the per-window arithmetic of the synthesisable sensor bridge in
``hdl/sensors/adc_to_spike_quantiser.v`` and the cycle-stepped golden model in
``tools/adc_to_spike_reference.py`` (Indiveri 2003 rate coding); the cycle-accurate
handshake/drain FSM stays in that reference, while this kernel is the hot
per-window compute.

The whole path is exact integer arithmetic (sign-aware Q-format rounding,
truncate-toward-zero window averaging, floor-division rate code), so the Python
floor and the Rust, Julia, Go and Mojo backends agree bit-for-bit; the parity
tolerance is exactly zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS
from sc_neurocore.accel.backend_selection import select_backend_order

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping


@dataclass(frozen=True)
class ADCSpikeWindowConfig:
    """Fixed-point and decimation contract for the ADC-to-spike encoder.

    Attributes
    ----------
    adc_width : int
        Raw ADC sample width in bits (must exceed one).
    q_int : int
        Q-format integer bits (must be positive).
    q_frac : int
        Q-format fractional bits (must be non-negative).
    decimation : int
        Number of ADC samples averaged into one spike window (must be positive).
    signed_input : bool
        ``True`` if the ADC delivers two's-complement samples, ``False`` for
        offset-binary samples centred at mid-scale.
    threshold_q : int
        Q-format magnitude that emits one spike (must be positive).
    """

    adc_width: int = 16
    q_int: int = 8
    q_frac: int = 8
    decimation: int = 8
    signed_input: bool = True
    threshold_q: int = 256

    @property
    def q_total(self) -> int:
        """Total Q-format bit width."""
        return self.q_int + self.q_frac

    @property
    def q_min(self) -> int:
        """Most negative representable Q-format code."""
        return -(1 << (self.q_total - 1))

    @property
    def q_max(self) -> int:
        """Most positive representable Q-format code."""
        return (1 << (self.q_total - 1)) - 1

    def validate(self) -> None:
        """Raise :class:`ValueError` if any field is out of contract.

        Raises
        ------
        ValueError
            If the ADC width, Q-format, decimation or threshold are invalid.
        """
        if self.adc_width <= 1:
            raise ValueError("adc_width must be greater than one")
        if self.q_int <= 0 or self.q_frac < 0:
            raise ValueError("Q-format needs positive integer bits and non-negative fraction bits")
        if self.decimation <= 0:
            raise ValueError("decimation must be positive")
        if self.threshold_q <= 0:
            raise ValueError("threshold_q must be positive")


@dataclass(frozen=True)
class ADCSpikeWindowResult:
    """Per-window outputs of the ADC-to-spike encoder.

    Each array is indexed by completed decimation window.

    Attributes
    ----------
    window_values_q : numpy.ndarray
        Sign-aware averaged Q-format window codes, ``int32``.
    spike_counts : numpy.ndarray
        Deterministic per-window spike counts (``|window| // threshold``),
        ``int32``.
    polarities : numpy.ndarray
        ``True`` where the window code is negative, ``bool_``.
    """

    window_values_q: npt.NDArray[np.int32]
    spike_counts: npt.NDArray[np.int32]
    polarities: npt.NDArray[np.bool_]


def quantise_adc(sample: int, config: ADCSpikeWindowConfig) -> int:
    """Centre and quantise one raw ADC sample to a Q-format code.

    Mirrors ``ADCToSpikeReference.quantise_adc``: two's-complement or offset-binary
    centring, Q-format up-shift or sign-aware round-down, then saturation.

    Parameters
    ----------
    sample : int
        Raw ADC sample.
    config : ADCSpikeWindowConfig
        Fixed-point contract.

    Returns
    -------
    int
        Saturated Q-format code.
    """
    adc_width = config.adc_width
    q_total = config.q_total
    if config.signed_input:
        sign_bit = 1 << (adc_width - 1)
        mask = (1 << adc_width) - 1
        sample &= mask
        centred = sample - (1 << adc_width) if sample & sign_bit else sample
    else:
        centred = sample - (1 << (adc_width - 1))

    if q_total > adc_width:
        rounded = centred << (q_total - adc_width)
    elif adc_width > q_total:
        shift = adc_width - q_total
        half = 1 << (shift - 1)
        rounded = (centred + half) >> shift if centred >= 0 else (centred - half) >> shift
    else:
        rounded = centred
    return max(config.q_min, min(config.q_max, rounded))


def _average_window(total_q: int, config: ADCSpikeWindowConfig) -> int:
    """Sign-aware round-then-truncate window average, mirroring the golden model."""
    half = config.decimation // 2
    adjusted = total_q + half if total_q >= 0 else total_q - half
    averaged = int(adjusted / config.decimation)
    return max(config.q_min, min(config.q_max, averaged))


def adc_to_spike_windows_q(
    samples: npt.ArrayLike,
    config: ADCSpikeWindowConfig | None = None,
) -> ADCSpikeWindowResult:
    """Pure-Python ADC-to-spike window encoder — the bit-true floor reference.

    Parameters
    ----------
    samples : array_like
        Raw ADC samples; the first ``n_windows * decimation`` are consumed.
    config : ADCSpikeWindowConfig, optional
        Fixed-point/decimation contract (defaults to Q8.8, decimation 8).

    Returns
    -------
    ADCSpikeWindowResult
        Per-window averaged codes, spike counts and polarities.

    Raises
    ------
    ValueError
        If the config is invalid or fewer than ``decimation`` samples are given.
    """
    cfg = config if config is not None else ADCSpikeWindowConfig()
    cfg.validate()
    sample_arr = np.ascontiguousarray(samples, dtype=np.int64).reshape(-1)
    n_windows = int(sample_arr.size) // cfg.decimation
    if n_windows == 0:
        raise ValueError(
            f"need at least decimation={cfg.decimation} samples, got {sample_arr.size}"
        )

    window_values = np.empty(n_windows, dtype=np.int32)
    spike_counts = np.empty(n_windows, dtype=np.int32)
    polarities = np.empty(n_windows, dtype=np.bool_)
    for window in range(n_windows):
        base = window * cfg.decimation
        total = 0
        for offset in range(cfg.decimation):
            total += quantise_adc(int(sample_arr[base + offset]), cfg)
        window_q = _average_window(total, cfg)
        window_values[window] = window_q
        spike_counts[window] = abs(window_q) // cfg.threshold_q
        polarities[window] = window_q < 0

    return ADCSpikeWindowResult(
        window_values_q=window_values,
        spike_counts=spike_counts,
        polarities=polarities,
    )


def _config_tuple(config: ADCSpikeWindowConfig) -> tuple[int, int, int, int, int, int]:
    """Flatten the config into the positional tuple the FFI backends consume."""
    return (
        config.adc_width,
        config.q_int,
        config.q_frac,
        config.decimation,
        1 if config.signed_input else 0,
        config.threshold_q,
    )


def _result_from_mapping(payload: Mapping[str, npt.ArrayLike]) -> ADCSpikeWindowResult:
    """Convert a backend dict payload into a typed :class:`ADCSpikeWindowResult`."""
    return ADCSpikeWindowResult(
        window_values_q=np.ascontiguousarray(payload["window_values_q"], dtype=np.int32),
        spike_counts=np.ascontiguousarray(payload["spike_counts"], dtype=np.int32),
        polarities=np.ascontiguousarray(payload["polarities"], dtype=np.bool_),
    )


def _backend_python(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
    return adc_to_spike_windows_q(samples, config)


def _backend_rust(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
    from sc_neurocore_engine import py_adc_to_spike_windows

    payload = py_adc_to_spike_windows(
        np.ascontiguousarray(samples, dtype=np.int64).reshape(-1), *_config_tuple(config)
    )
    return _result_from_mapping(payload)


def _backend_julia(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
    from sc_neurocore.accel.julia.adc_to_spike import adc_to_spike_windows as julia_windows

    return _result_from_mapping(julia_windows(samples, *_config_tuple(config)))


def _backend_go(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
    from sc_neurocore.accel.go.adc_to_spike import adc_to_spike_windows as go_windows

    return _result_from_mapping(go_windows(samples, *_config_tuple(config)))


def _backend_mojo(samples: npt.ArrayLike, config: ADCSpikeWindowConfig) -> ADCSpikeWindowResult:
    from sc_neurocore.accel.mojo.adc_to_spike import adc_to_spike_windows as mojo_windows

    return _result_from_mapping(mojo_windows(samples, *_config_tuple(config)))


_BACKEND_DISPATCH: dict[
    str, Callable[[npt.ArrayLike, ADCSpikeWindowConfig], ADCSpikeWindowResult]
] = {
    "python": _backend_python,
    "rust": _backend_rust,
    "julia": _backend_julia,
    "go": _backend_go,
    "mojo": _backend_mojo,
}


def available_backends() -> dict[str, bool]:
    """Probe which acceleration backends can run the ADC-to-spike kernel.

    Returns
    -------
    dict
        Mapping of backend name to availability, in fastest-first order. The
        ``python`` floor is always ``True``.
    """
    status: dict[str, bool] = {}
    probe_samples = np.zeros(ADCSpikeWindowConfig().decimation, dtype=np.int64)
    probe_config = ADCSpikeWindowConfig()
    for name in FASTEST_FIRST_BACKENDS:
        if name == "python":
            status[name] = True
            continue
        try:
            _BACKEND_DISPATCH[name](probe_samples, probe_config)
            status[name] = True
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            status[name] = False
    return status


def adc_to_spike_windows(
    samples: npt.ArrayLike,
    config: ADCSpikeWindowConfig | None = None,
    *,
    backend: str = "auto",
) -> ADCSpikeWindowResult:
    """Encode ADC samples into spike windows through the fastest available backend.

    Parameters
    ----------
    samples : array_like
        Raw ADC samples.
    config : ADCSpikeWindowConfig, optional
        Fixed-point/decimation contract.
    backend : str, optional
        ``"auto"`` (default) selects the fastest available backend in
        :data:`FASTEST_FIRST_BACKENDS` order; a specific name forces that backend.

    Returns
    -------
    ADCSpikeWindowResult
        Bit-identical to the Python floor for every backend.

    Raises
    ------
    ValueError
        If ``backend`` is not a known name.
    ImportError
        If an explicitly requested accelerator backend is unavailable.
    """
    cfg = config if config is not None else ADCSpikeWindowConfig()
    if backend != "auto":
        if backend not in _BACKEND_DISPATCH:
            raise ValueError(
                f"unknown backend {backend!r}; choose from {('auto', *FASTEST_FIRST_BACKENDS)}"
            )
        return _BACKEND_DISPATCH[backend](samples, cfg)
    for name in select_backend_order("adc_to_spike_windows_q"):
        if name == "python":
            break
        try:
            return _BACKEND_DISPATCH[name](samples, cfg)
        except (ImportError, OSError, RuntimeError, FileNotFoundError):
            continue
    return _backend_python(samples, cfg)


__all__ = [
    "FASTEST_FIRST_BACKENDS",
    "ADCSpikeWindowConfig",
    "ADCSpikeWindowResult",
    "adc_to_spike_windows",
    "adc_to_spike_windows_q",
    "available_backends",
    "quantise_adc",
]
