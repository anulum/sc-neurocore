# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Hindmarsh-Rose bit-exact backend tests

"""Bit-exact Rust, Julia, and Go Hindmarsh–Rose parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tests.hindmarsh_rose_backends_support import _BIT_EXACT, _CURRENTS, _run


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    ref_trace, ref_spikes, rx, ry, rz = _run("python", current=current)
    trace, spikes, xf, yf, zf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (xf, yf, zf) == (rx, ry, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rx, ry, rz = _run("python", n=n)
        got, gs, gx, gy, gz = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gx, gy, gz) == (rs, rx, ry, rz)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_chaotic_long_run(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    # Exact arithmetic stays bit-exact even across a long chaotic bursting run,
    # where a transcendental or FMA backend would have diverged completely.
    if not available():
        pytest.skip(f"{backend} Hindmarsh-Rose backend unavailable")
    ref, rs, rx, ry, rz = _run("python", current=3.0, n=60000)
    got, gs, gx, gy, gz = _run(backend, current=3.0, n=60000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gx, gy, gz) == (rs, rx, ry, rz)
