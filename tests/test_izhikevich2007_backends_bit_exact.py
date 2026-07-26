# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Izhikevich-2007 bit-exact backend tests

"""Bit-exact Rust, Julia, and Go Izhikevich-2007 parity contracts."""

from __future__ import annotations

import numpy as np
import pytest

from tests.izhikevich2007_backends_support import _BIT_EXACT, _CURRENTS, _run


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
@pytest.mark.parametrize("current", _CURRENTS)
def test_bit_exact_trace(backend: str, available, current: float) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    ref_trace, ref_spikes, rv, ru = _run("python", current=current)
    trace, spikes, vf, uf = _run(backend, current=current)
    np.testing.assert_array_equal(trace, ref_trace)
    assert spikes == ref_spikes
    assert (vf, uf) == (rv, ru)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_empty_and_single(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    for n in (0, 1, 2):
        ref, rs, rv, ru = _run("python", n=n)
        got, gs, gv, gu = _run(backend, n=n)
        np.testing.assert_array_equal(got, ref)
        assert (gs, gv, gu) == (rs, rv, ru)


@pytest.mark.parametrize("backend,available", _BIT_EXACT, ids=[b for b, _ in _BIT_EXACT])
def test_bit_exact_tonic_long_run(backend: str, available) -> None:  # type: ignore[no-untyped-def] # Preserved legacy test AST
    # A long tonic-spiking run with many threshold resets stays bit-exact — the
    # exact RHS and the reset have no order sensitivity.
    if not available():
        pytest.skip(f"{backend} Izhikevich2007 backend unavailable")
    ref, rs, rv, ru = _run("python", current=300.0, n=60000)
    got, gs, gv, gu = _run(backend, current=300.0, n=60000)
    np.testing.assert_array_equal(got, ref)
    assert (gs, gv, gu) == (rs, rv, ru)
