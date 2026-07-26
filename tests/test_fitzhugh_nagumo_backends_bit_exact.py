# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — FitzHugh-Nagumo bit-exact backend tests

"""Bit-exact Rust, Julia, and Go FitzHugh–Nagumo parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tests.fitzhugh_nagumo_backends_support import _BIT_EXACT, _CURRENTS, _run


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    ref_trace, ref_spikes, rv, rw = _run("python", current=current)
    trace, spikes, vf, wf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert vf == rv and wf == rw


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, rw = _run("python", n=n)
        got, gs, gv, gw = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gw) == (rs, rv, rw)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_limit_cycle_long_run(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    # A sustained limit cycle (I = 0.5) over a long horizon stays bit-exact —
    # the exact RHS has no order sensitivity and the 2-D flow cannot diverge.
    if not available():
        pytest.skip(f"{backend} FitzHugh-Nagumo backend unavailable")
    ref, rs, rv, rw = _run("python", current=0.5, n=50000)
    got, gs, gv, gw = _run(backend, current=0.5, n=50000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gw) == (rs, rv, rw)
