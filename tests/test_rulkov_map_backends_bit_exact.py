# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Focused Rulkov backend contracts

"""Focused cross-backend Rulkov map contracts."""

from .rulkov_map_backends_support import *


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("sigma", _REGIMES)
def test_bit_exact_trace(backend: str, available, sigma: float) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Rulkov backend unavailable")
    ref_trace, ref_spikes, rx, ry = _run("python", sigma=sigma)
    trace, spikes, xf, yf = _run(backend, sigma=sigma)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert xf == rx and yf == ry


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Rulkov backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry = _run("python", n=n)
        got, gs, gx, gy = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy) == (rs, rx, ry)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_high_current_spiking(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    # High constant drive exercises branches 2 and 3 (plateau + hard reset).
    if not available():
        pytest.skip(f"{backend} Rulkov backend unavailable")
    ref, rs, rx, ry = _run("python", current=5.0, n=8000)
    got, gs, gx, gy = _run(backend, current=5.0, n=8000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx, gy) == (rs, rx, ry)
