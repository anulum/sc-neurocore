# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — ADC-to-spike encoder — cross-language + golden parity tests

"""Bit-exact parity of every backend against the Python floor.

The Python floor is also checked against the cycle-stepped golden model in
``tools/adc_to_spike_reference.py``.

The integer per-window encode is exact, so the cross-language contract is identical
raw arrays (tolerance zero). Each backend test skips with an explanatory reason
when its toolchain artefact is not built.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.testing as npt
import numpy.typing as nptyping
import pytest

from sc_neurocore.sensors.adc_to_spike_kernel import (
    ADCSpikeWindowConfig,
    ADCSpikeWindowResult,
    adc_to_spike_windows,
    adc_to_spike_windows_q,
    available_backends,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from adc_to_spike_reference import ADCSpikeConfig, ADCToSpikeReference  # noqa: E402

_BACKENDS = ("rust", "julia", "go", "mojo")


@lru_cache(maxsize=1)
def _availability() -> dict[str, bool]:
    return available_backends()


_RESULT_FIELDS = ("window_values_q", "spike_counts", "polarities")


def _assert_bit_exact(reference: ADCSpikeWindowResult, candidate: ADCSpikeWindowResult) -> None:
    for field in _RESULT_FIELDS:
        npt.assert_array_equal(getattr(reference, field), getattr(candidate, field), err_msg=field)


def _named_configs() -> dict[str, ADCSpikeWindowConfig]:
    return {
        "default_q8_8": ADCSpikeWindowConfig(),
        "offset_binary": ADCSpikeWindowConfig(signed_input=False, decimation=4, threshold_q=128),
        "narrow_adc_up_shift": ADCSpikeWindowConfig(adc_width=12, q_int=8, q_frac=8, decimation=16),
        "wide_adc_round_down": ADCSpikeWindowConfig(adc_width=20, q_int=8, q_frac=8),
        "low_threshold_q4_4": ADCSpikeWindowConfig(adc_width=16, q_int=4, q_frac=4, threshold_q=16),
    }


def _samples(
    config: ADCSpikeWindowConfig, n_windows: int, seed: int
) -> nptyping.NDArray[np.int64]:
    rng = np.random.default_rng(seed)
    return rng.integers(
        0, 1 << config.adc_width, size=config.decimation * n_windows, dtype=np.int64
    )


@pytest.mark.parametrize("backend", _BACKENDS)
@pytest.mark.parametrize("config_name", list(_named_configs()))
def test_backend_bit_exact_with_python(backend: str, config_name: str) -> None:
    """Built accelerator backends match the Python floor with zero tolerance."""
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    config = _named_configs()[config_name]
    samples = _samples(config, 200, seed=7)
    reference = adc_to_spike_windows_q(samples, config)
    candidate = adc_to_spike_windows(samples, config, backend=backend)
    _assert_bit_exact(reference, candidate)


@pytest.mark.parametrize("config_name", list(_named_configs()))
def test_python_floor_matches_golden_reference(config_name: str) -> None:
    """The batched Python floor matches the cycle-stepped golden reference."""
    config = _named_configs()[config_name]
    samples = _samples(config, 120, seed=11)
    reference = ADCToSpikeReference(
        ADCSpikeConfig(
            adc_width=config.adc_width,
            q_int=config.q_int,
            q_frac=config.q_frac,
            decimation=config.decimation,
            signed_input=config.signed_input,
            threshold_q=config.threshold_q,
        )
    )
    n_windows = samples.size // config.decimation
    expected_windows: list[int] = []
    expected_counts: list[int] = []
    expected_polarities: list[bool] = []
    for window in range(n_windows):
        base = window * config.decimation
        total = sum(
            reference.quantise_adc(int(samples[base + offset]))
            for offset in range(config.decimation)
        )
        window_q = reference._average_window(total)
        expected_windows.append(window_q)
        expected_counts.append(abs(window_q) // config.threshold_q)
        expected_polarities.append(window_q < 0)

    result = adc_to_spike_windows_q(samples, config)
    npt.assert_array_equal(result.window_values_q, expected_windows)
    npt.assert_array_equal(result.spike_counts, expected_counts)
    npt.assert_array_equal(result.polarities, expected_polarities)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_rejects_short_stream(backend: str) -> None:
    """Built accelerator backends reject streams shorter than one decimation window."""
    if not _availability().get(backend, False):
        pytest.skip(f"{backend} backend not built in this environment")
    with pytest.raises((ValueError, RuntimeError)):
        adc_to_spike_windows([1, 2, 3], ADCSpikeWindowConfig(decimation=8), backend=backend)


def test_at_least_python_backend_present() -> None:
    """The Python floor backend is always available for production fallback."""
    assert _availability()["python"] is True
