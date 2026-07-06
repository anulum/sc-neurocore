# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Data-driven backend selector contract tests

"""Contract for the benchmark-driven polyglot backend ordering.

Tests pin an explicit recorded-host CPU string so the measured path is exercised
deterministically on any runner, plus the unknown-host/kernel fallbacks. The
recorded benchmarks live in ``benchmarks/results/`` and are read read-only.
"""

from __future__ import annotations

from pathlib import Path
import platform
from typing import Protocol, cast

import numpy.testing as npt
import pytest

from sc_neurocore.accel import backend_selection as bs
from sc_neurocore.accel.backend_order import FASTEST_FIRST_BACKENDS

#: CPU model recorded in the committed per-backend benchmark JSONs.
_RECORDED_CPU = "11th Gen Intel(R) Core(TM) i5-11600K @ 3.90GHz"
_DCLS = "dcls_max_forward_batch_q88"
_ADC = "adc_to_spike_windows_q"
_MIXED = "mixed_dense_forward_batch_q88_q1616"


class _MeasuredOrdersCache(Protocol):
    """Protocol for the cache controls installed by ``functools.cache``."""

    def cache_clear(self) -> None:
        """Clear the cached benchmark-order table."""


def _clear_measured_orders_cache() -> None:
    """Clear benchmark-order cache entries after path monkeypatching."""
    cast(_MeasuredOrdersCache, bs.measured_orders).cache_clear()


def test_current_cpu_is_nonempty_string() -> None:
    cpu = bs.current_cpu()
    assert isinstance(cpu, str) and cpu


def test_current_cpu_uses_platform_processor_without_proc_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host detection falls back to ``platform.processor`` without model-name data."""

    def read_cpuinfo_without_model(
        _path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        assert encoding == "utf-8"
        assert errors is None
        return "processor\t: 0\nvendor_id\t: GenuineIntel\n"

    monkeypatch.setattr(Path, "read_text", read_cpuinfo_without_model)
    monkeypatch.setattr(platform, "processor", lambda: "portable-cpu")

    assert bs.current_cpu() == "portable-cpu"


def test_current_cpu_returns_unknown_when_proc_and_platform_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host detection keeps a deterministic fallback when CPU probes fail."""

    def raise_cpuinfo_oserror(
        _path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        assert encoding == "utf-8"
        assert errors is None
        raise OSError("cpuinfo unavailable")

    monkeypatch.setattr(Path, "read_text", raise_cpuinfo_oserror)
    monkeypatch.setattr(platform, "processor", lambda: "")

    assert bs.current_cpu() == "unknown"


def test_measured_orders_indexes_recorded_host_and_kernels() -> None:
    table = bs.measured_orders()
    assert _RECORDED_CPU in table
    kernels = table[_RECORDED_CPU]
    for kernel in (_DCLS, _ADC, _MIXED):
        assert kernel in kernels
        assert kernels[kernel][-1] == "python" or "python" in kernels[kernel]


def test_backend_speed_order_filters_unavailable_and_sorts_ascending() -> None:
    backends = {
        "rust": {"available": True, "used": True, "median_call_ms": 1.5},
        "go": {"available": True, "used": True, "median_call_ms": 0.5},
        "mojo": {"available": False, "used": False, "median_call_ms": 0.1},
        "julia": {"available": True, "used": True, "median_call_ms": None},
        "python": {"available": True, "used": True, "median_call_ms": 9.0},
        "stray": "not-a-dict",
    }
    assert bs._backend_speed_order(backends) == ["go", "rust", "python"]


def test_measured_orders_returns_empty_when_results_dir_is_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Missing benchmark-results directories leave the static fallback in force."""
    monkeypatch.setattr(bs, "_RESULTS_DIR", tmp_path / "missing-results")
    _clear_measured_orders_cache()

    assert bs.measured_orders() == {}

    _clear_measured_orders_cache()


def test_measured_orders_skips_malformed_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Malformed benchmark files are ignored rather than breaking dispatch."""
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    monkeypatch.setattr(bs, "_RESULTS_DIR", tmp_path)
    _clear_measured_orders_cache()

    assert bs.measured_orders() == {}

    _clear_measured_orders_cache()


def test_select_uses_measured_order_for_recorded_host() -> None:
    # On the recorded host rust leads dcls but julia beat mojo — a real reordering
    # versus the static ("rust","mojo","julia","go","python").
    assert bs.select_backend_order(_DCLS, cpu=_RECORDED_CPU) == (
        "rust",
        "julia",
        "mojo",
        "go",
        "python",
    )


def test_measured_winner_differs_per_kernel() -> None:
    # The whole point of being data-driven: the fastest backend is kernel-specific,
    # not the statically-assumed "rust first" for every kernel.
    assert bs.select_backend_order(_ADC, cpu=_RECORDED_CPU)[0] == "mojo"
    assert bs.select_backend_order(_MIXED, cpu=_RECORDED_CPU)[0] == "julia"


def test_unknown_kernel_falls_back_to_static() -> None:
    assert bs.select_backend_order("no_such_kernel", cpu=_RECORDED_CPU) == FASTEST_FIRST_BACKENDS


def test_unknown_host_falls_back_to_static() -> None:
    assert bs.select_backend_order(_DCLS, cpu="Imaginary CPU 9000") == FASTEST_FIRST_BACKENDS


def test_floor_last_and_backend_set_preserved() -> None:
    for kernel in (_DCLS, _ADC, _MIXED):
        order = bs.select_backend_order(kernel, cpu=_RECORDED_CPU)
        assert order[-1] == "python"
        assert set(order) == set(FASTEST_FIRST_BACKENDS)


def test_custom_numpy_floor_static_is_respected() -> None:
    # Matches accel/backend.py PRIORITY (numpy floor). The non-static "python"
    # measurement is dropped; numpy stays the final tier.
    static = ("rust", "mojo", "julia", "go", "numpy")
    order = bs.select_backend_order(_DCLS, static=static, cpu=_RECORDED_CPU)
    assert order == ("rust", "julia", "mojo", "go", "numpy")


def test_empty_static_returns_empty() -> None:
    assert bs.select_backend_order(_DCLS, static=(), cpu=_RECORDED_CPU) == ()


def test_auto_dispatch_matches_python_floor_after_reorder() -> None:
    # Reordering must not change results: the selector-driven "auto" path is
    # bit-identical to the always-available Python floor.
    from sc_neurocore.scpn.dcls_tent_kernel import dcls_max_forward_batch

    spikes = [1, 1, 1, 0, 1, 0]
    weights = [256, 128, -64, 512, -256, 64]
    centres = [256, 512]
    sigmas = [512, 768]
    n_taps = 3
    auto = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="auto")
    floor = dcls_max_forward_batch(spikes, weights, centres, sigmas, n_taps, backend="python")
    npt.assert_array_equal(auto.outputs_q88, floor.outputs_q88)
    npt.assert_array_equal(auto.accumulators_q16_16, floor.accumulators_q16_16)
