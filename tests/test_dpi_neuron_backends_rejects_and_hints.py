# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_hints) from former test_dpi_neuron_backends.py

from __future__ import annotations

from tests.dpi_neuron_backends_support import *  # noqa: F403


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_c_abi_rejects_invalid_run_without_writing_output(backend: str) -> None:
    """Reject invalid work before emitting any caller-visible state."""
    assert getattr(backends, f"ensure_{backend}_loaded")()
    output = np.full(4, -999.0, dtype=np.float64)
    values = (*_factory_values(), 1, math.nan)
    if backend == "go":
        assert backends._go_lib is not None
        result = backends._go_lib.dpi_neuron_simulate_c(
            *values, output.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        )
    else:
        assert backends._mojo_lib is not None
        result = backends._mojo_lib.dpi_neuron_simulate_c(*values, int(output.ctypes.data))
    assert result == -1
    np.testing.assert_array_equal(output, np.full(4, -999.0, dtype=np.float64))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_complete_c_abi_rejects_without_writing_any_buffer(backend: str) -> None:
    """Keep every state and event buffer untouched after complete-run rejection."""
    assert getattr(backends, f"ensure_{backend}_loaded")()
    state_outputs = [np.full(2, -999.0, dtype=np.float64) for _ in range(3)]
    event_output = np.full(1, 255, dtype=np.uint8)
    values = (*_factory_values(), 1, math.nan)
    library = getattr(backends, f"_{backend}_lib")
    assert library is not None
    if backend == "go":
        destinations: tuple[object, ...] = (
            *(output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)) for output in state_outputs),
            event_output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        )
    else:
        destinations = (
            *(int(output.ctypes.data) for output in state_outputs),
            int(event_output.ctypes.data),
        )
    result = library.dpi_neuron_simulate_complete_c(*values, *destinations)
    assert result == -1
    for output in state_outputs:
        np.testing.assert_array_equal(output, np.full(2, -999.0, dtype=np.float64))
    np.testing.assert_array_equal(event_output, np.full(1, 255, dtype=np.uint8))


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_native_overflow_rejection_does_not_commit_instance_state(backend: str) -> None:
    """Translate native pre-reset overflow into mutation-free public failure."""
    neuron = DPINeuron(tau=sys.float_info.min)
    before = (neuron.i_mem, neuron.i_ahp, neuron.refractory_time)
    with pytest.raises(FloatingPointError, match="kernel rejected"):
        neuron.simulate(1, sys.float_info.max, backend=backend)
    assert (neuron.i_mem, neuron.i_ahp, neuron.refractory_time) == before


@pytest.mark.parametrize("backend", _FULL_CONTRACT_BACKENDS)
def test_requested_backend_reports_unavailable(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Return an actionable failure instead of silently falling back."""
    monkeypatch.setattr(backends, f"ensure_{backend}_loaded", lambda: False)
    with pytest.raises(RuntimeError, match=backend.title()):
        DPINeuron().simulate(1, 0.0, backend=backend)


def test_requested_rust_backend_reports_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep explicit Rust selection fail-closed when the extension is absent."""
    monkeypatch.setattr(backends, "_HAS_RUST", False)
    monkeypatch.setattr(backends, "_EngineDPICls", None)
    with pytest.raises(RuntimeError, match="Rust DPI"):
        DPINeuron().simulate(1, 0.0, backend="rust")


def test_go_build_hint_uses_reproducible_package_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Build the public Go hint without changing the committed C header."""
    monkeypatch.setattr(backends, "ensure_go_loaded", lambda: False)
    with pytest.raises(RuntimeError) as error:
        DPINeuron().simulate(1, 0.0, backend="go")
    expected_command = "go build -buildmode=c-shared -o libdpi_neuron.so ."
    assert expected_command in str(error.value)

    source = _REPOSITORY / "src" / "sc_neurocore" / "accel" / "go" / "neurons" / "dpi_neuron"
    output = tmp_path / "libdpi_neuron.so"
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(output), "."],
        cwd=source,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    generated_header = output.with_suffix(".h")
    assert generated_header.read_bytes() == (source / "libdpi_neuron.h").read_bytes()


def test_dispatcher_runners_reject_missing_loaded_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Protect direct runner calls as well as the public model boundary."""
    monkeypatch.setattr(backends, "_EngineDPICls", None)
    with pytest.raises(RuntimeError, match="Rust DPI engine"):
        backends.simulate_rust(1, 0.0)
    monkeypatch.setattr(backends, "_EngineDPICompleteFn", None)
    with pytest.raises(RuntimeError, match="Rust DPI complete"):
        _invoke_full_contract(backends.simulate_rust_complete)
    monkeypatch.setattr(backends, "_julia_module", None)
    with pytest.raises(RuntimeError, match="Julia DPI module"):
        _invoke_full_contract(backends.simulate_julia)
    with pytest.raises(RuntimeError, match="Julia DPI module"):
        _invoke_full_contract(backends.simulate_julia_complete)
    monkeypatch.setattr(backends, "_go_lib", None)
    with pytest.raises(RuntimeError, match="Go DPI library"):
        _invoke_full_contract(backends.simulate_go)
    with pytest.raises(RuntimeError, match="Go DPI library"):
        _invoke_full_contract(backends.simulate_go_complete)
    monkeypatch.setattr(backends, "_mojo_lib", None)
    with pytest.raises(RuntimeError, match="Mojo DPI library"):
        _invoke_full_contract(backends.simulate_mojo)
    with pytest.raises(RuntimeError, match="Mojo DPI library"):
        _invoke_full_contract(backends.simulate_mojo_complete)


def test_direct_c_runner_rejection_names_backend() -> None:
    """Use distinct actionable errors for Go and Mojo ABI rejection."""

    class RejectingLibrary:
        def dpi_neuron_simulate_c(self, *_args: object) -> int:
            return -1

    values = _factory_values()
    with pytest.raises(FloatingPointError, match="Go DPI"):
        backends._simulate_c(RejectingLibrary(), values, 1, 0.0, mojo=False)
    with pytest.raises(FloatingPointError, match="Mojo DPI"):
        backends._simulate_c(RejectingLibrary(), values, 1, 0.0, mojo=True)


@pytest.mark.parametrize("mojo", (False, True))
@pytest.mark.parametrize("reported_events", (-1, 1))
def test_complete_c_runner_rejects_invalid_event_accounting(
    mojo: bool,
    reported_events: int,
) -> None:
    """Reject native packets that fail or disagree with their event vector."""

    class RejectingLibrary:
        def dpi_neuron_simulate_complete_c(self, *_args: object) -> int:
            return reported_events

    backend = "Mojo" if mojo else "Go"
    with pytest.raises(FloatingPointError, match=f"{backend} DPI complete"):
        backends._simulate_c_complete(
            RejectingLibrary(),
            _factory_values(),
            0,
            0.0,
            mojo=mojo,
        )


def test_legacy_dispatchers_preserve_their_compatibility_packets() -> None:
    """Exercise the retained one-trace interfaces for every compiled lane."""
    assert backends.ensure_julia_loaded()
    assert backends.ensure_go_loaded()
    assert backends.ensure_mojo_loaded()
    rust = backends.simulate_rust(3, 0.0)
    julia = _invoke_full_contract(backends.simulate_julia)
    go = _invoke_full_contract(backends.simulate_go)
    mojo = _invoke_full_contract(backends.simulate_mojo)
    for (trace, spikes, state), expected_length in zip(
        (rust, julia, go, mojo),
        (3, 1, 1, 1),
        strict=True,
    ):
        assert trace.shape == (expected_length,)
        assert spikes == 0
        assert len(state) == 3


def test_rust_complete_normalises_byte_and_array_event_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accept both PyO3 byte buffers and already materialised event arrays."""
    prefix = (np.zeros(1), np.zeros(1), np.zeros(1))

    for raw_events in (b"\x01", np.asarray([1], dtype=np.uint8)):
        monkeypatch.setattr(
            backends,
            "_EngineDPICompleteFn",
            lambda *_args, events=raw_events: (*prefix, events, 0.1, 0.2, 0.3),
        )
        result = _invoke_full_contract(backends.simulate_rust_complete)
        np.testing.assert_array_equal(result[3], np.asarray([1], dtype=np.uint8))
        assert result[4] == (0.1, 0.2, 0.3)
