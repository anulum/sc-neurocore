# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused backend selection contracts

"""Focused data-driven backend selection contracts."""

from .backend_selection_support import *


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
