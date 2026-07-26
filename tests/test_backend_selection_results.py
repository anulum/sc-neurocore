# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused backend selection contracts

"""Focused data-driven backend selection contracts."""

from .backend_selection_support import *


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
