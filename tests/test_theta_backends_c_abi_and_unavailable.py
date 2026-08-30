# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (c_abi_and_unavailable) from former test_theta_backends.py

from __future__ import annotations

from tests.theta_backends_support import *  # noqa: F403


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_run_without_writing_output(backend: str) -> None:
    """Reject invalid work before emitting any caller-visible phase."""
    assert getattr(backends, f"ensure_{backend}_loaded")()
    output = np.full(2, -999.0, dtype=np.float64)
    if backend == "go":
        assert backends._go_lib is not None
        result = backends._go_lib.theta_simulate_c(
            0.25,
            0.01,
            1,
            math.nan,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    else:
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.theta_simulate_c(
            0.25, 0.01, 1, math.nan, int(output.ctypes.data)
        )
    assert result == -1
    np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_complete_c_abi_rejects_without_writing_either_buffer(backend: str) -> None:
    """Keep both caller-owned buffers untouched on complete-packet rejection."""
    assert getattr(backends, f"ensure_{backend}_loaded")()
    phase = np.full(2, -999.0, dtype=np.float64)
    events = np.full(1, 255, dtype=np.uint8)
    library = getattr(backends, f"_{backend}_lib")
    assert library is not None
    if backend == "go":
        result = library.theta_simulate_complete_c(
            0.25,
            0.01,
            1,
            math.nan,
            phase.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            events.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    else:
        result = library.theta_simulate_complete_c(
            0.25,
            0.01,
            1,
            math.nan,
            int(phase.ctypes.data),
            int(events.ctypes.data),
        )
    assert result == -1
    np.testing.assert_array_equal(phase, np.full(2, -999.0, dtype=np.float64))
    np.testing.assert_array_equal(events, np.full(1, 255, dtype=np.uint8))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate a native non-finite candidate into mutation-free failure."""
    neuron = ThetaNeuron(theta=0.25, dt=1.0e308)
    before = neuron.theta
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, -1.0e308, backend=backend)
    assert neuron.theta == before


@pytest.mark.parametrize("backend", ("julia", "go", "mojo"))
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back."""
    monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        ThetaNeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust requests fail-closed when the engine is absent."""
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    monkeypatch.setattr(backends, "_EngineThetaCls", None)
    with pytest.raises(RuntimeError, match="Rust Theta backend"):
        ThetaNeuron().simulate(1, 0.0, backend="rust")
