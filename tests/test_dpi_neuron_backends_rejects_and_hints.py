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
    monkeypatch.setattr(backends, "_julia_module", None)
    with pytest.raises(RuntimeError, match="Julia DPI module"):
        _invoke_full_contract(backends.simulate_julia)
    monkeypatch.setattr(backends, "_go_lib", None)
    with pytest.raises(RuntimeError, match="Go DPI library"):
        _invoke_full_contract(backends.simulate_go)
    monkeypatch.setattr(backends, "_mojo_lib", None)
    with pytest.raises(RuntimeError, match="Mojo DPI library"):
        _invoke_full_contract(backends.simulate_mojo)


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
