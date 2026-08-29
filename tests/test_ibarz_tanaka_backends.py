# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Ibarz-Tanaka 2007 polyglot parity tests

"""Cross-backend parity for the source-derived four-branch map."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from types import SimpleNamespace

import numpy as np
import pytest

from sc_neurocore.neurons.models import ibarz_tanaka_map
from sc_neurocore.neurons.models.ibarz_tanaka_map import IbarzTanakaMapNeuron


def _run(
    backend: str, *, current: float = 0.2, n_steps: int = 1000
) -> tuple[np.ndarray, int, float, float]:
    neuron = IbarzTanakaMapNeuron()
    trace, events = neuron.simulate(n_steps, current, backend=backend)
    return trace, events, neuron.v, neuron.u


def _rust() -> bool:
    return ibarz_tanaka_map._HAS_RUST


def _julia() -> bool:
    return ibarz_tanaka_map._ensure_julia_loaded()


def _go() -> bool:
    return ibarz_tanaka_map._ensure_go_loaded()


def _mojo() -> bool:
    return ibarz_tanaka_map._ensure_mojo_loaded()


_BIT_EXACT: tuple[tuple[str, Callable[[], bool]], ...] = (
    ("rust", _rust),
    ("julia", _julia),
    ("go", _go),
)


@pytest.mark.parametrize(("backend", "available"), _BIT_EXACT)
@pytest.mark.parametrize("current", (0.0, 0.2, 1.0))
def test_exact_backends_match_python(
    backend: str, available: Callable[[], bool], current: float
) -> None:
    """Rust, Julia and Go reproduce every committed fast-state bit."""
    if not available():
        pytest.skip(f"{backend} backend is not built")
    expected = _run("python", current=current)
    actual = _run(backend, current=current)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1:] == expected[1:]


@pytest.mark.parametrize("current", (0.0, 0.2, 1.0))
def test_mojo_matches_source_events_and_measured_error_bound(current: float) -> None:
    """Mojo preserves events and stays within the measured absolute-error bound."""
    if not _mojo():
        pytest.skip("mojo backend is not built")
    expected = _run("python", current=current)
    actual = _run("mojo", current=current)
    np.testing.assert_allclose(actual[0], expected[0], rtol=0.0, atol=1.5e-8)
    assert actual[1] == expected[1]
    assert actual[2] == pytest.approx(expected[2], abs=1.5e-8)
    assert actual[3] == pytest.approx(expected[3], abs=1.5e-8)


def test_zero_step_contract_preserves_state_across_available_backends() -> None:
    """The empty batch returns an empty trace and preserves both states."""
    backends = [("python", lambda: True), *_BIT_EXACT, ("mojo", _mojo)]
    for backend, available in backends:
        if not available():
            continue
        trace, events, v_final, u_final = _run(backend, n_steps=0)
        assert trace.size == 0
        assert events == 0
        assert (v_final, u_final) == (-1.0, -0.1)


def test_explicit_unavailable_backend_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """An explicit unavailable backend cannot silently fall back to Python."""
    monkeypatch.setattr(ibarz_tanaka_map, "_HAS_RUST", False)
    with pytest.raises(RuntimeError, match="rust Ibarz-Tanaka backend is unavailable"):
        IbarzTanakaMapNeuron().simulate(1, 0.2, backend="rust")


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_negative_native_status_is_failure_atomic(
    monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """A rejected C-ABI batch cannot be committed as a negative event count."""

    def reject(*_args: object) -> int:
        return -2

    library = SimpleNamespace(ibarz_tanaka_map_simulate_c=reject)
    monkeypatch.setattr(ibarz_tanaka_map, f"_{backend}_lib", library)
    monkeypatch.setattr(ibarz_tanaka_map, f"_HAS_{backend.upper()}", True)
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)

    with pytest.raises(FloatingPointError, match=f"{backend.title()} Ibarz-Tanaka"):
        neuron.simulate(8, 0.2, backend=backend)
    assert (neuron.v, neuron.u) == before


def test_malformed_accelerator_packet_is_failure_atomic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Trace shape and final-state validation precede object mutation."""

    def malformed(*_args: object) -> tuple[list[float], int, float, float]:
        return [1.0], 0, 1.0, 2.0

    monkeypatch.setattr(ibarz_tanaka_map, "_rust_simulate", malformed)
    monkeypatch.setattr(ibarz_tanaka_map, "_HAS_RUST", True)
    neuron = IbarzTanakaMapNeuron()
    before = (neuron.v, neuron.u)

    with pytest.raises(RuntimeError, match="trace shape"):
        neuron.simulate(2, 0.2, backend="rust")
    assert (neuron.v, neuron.u) == before


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_real_c_abi_overflow_is_failure_atomic(backend: str) -> None:
    """The compiled C ABI validates the complete orbit before touching output."""
    available = _go() if backend == "go" else _mojo()
    if not available:
        pytest.skip(f"{backend} backend is not built")
    library = ibarz_tanaka_map._go_lib if backend == "go" else ibarz_tanaka_map._mojo_lib
    assert library is not None
    output = np.full(6, 123.5, dtype=np.float64)
    output_argument: object
    if backend == "go":
        output_argument = output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    else:
        output_argument = int(output.ctypes.data)

    status = library.ibarz_tanaka_map_simulate_c(
        1.0e308,
        1.0e308,
        1.0e308,
        1.0,
        0.0,
        4,
        0.0,
        output_argument,
    )
    assert status == -2
    np.testing.assert_array_equal(output, np.full(6, 123.5, dtype=np.float64))
