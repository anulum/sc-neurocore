# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - Focused free-test suite (rejects_and_reset) from former test_chialvo_map_backends.py

from __future__ import annotations

from tests.chialvo_map_backends_support import *  # noqa: F403


def test_invalid_backend_rejected() -> None:
    for backend in ("cuda", "", "RUST"):
        with pytest.raises(ValueError, match="backend must be"):
            ChialvoMapNeuron().simulate(1, backend=backend)


def test_invalid_batch_arguments_and_mutable_configuration_rejected() -> None:
    neuron = ChialvoMapNeuron()
    with pytest.raises(ValueError, match="non-negative"):
        neuron.simulate(-1)
    with pytest.raises(ValueError, match="current"):
        neuron.simulate(1, current=np.inf)
    neuron.k = np.nan
    with pytest.raises(ValueError, match="k"):
        neuron.simulate(1)


def test_complete_python_batch_is_failure_atomic() -> None:
    neuron = ChialvoMapNeuron(x=1.0e308, y=0.0)
    initial = (neuron.x, neuron.y)
    with pytest.raises(FloatingPointError, match="candidate state"):
        neuron.simulate_complete(2, backend="python")
    assert (neuron.x, neuron.y) == initial


@pytest.mark.parametrize("backend", ("go", "mojo"))
def test_complete_c_abi_failure_leaves_output_buffers_untouched(backend: str) -> None:
    available = _go_available if backend == "go" else _mojo_available
    _require(backend, available)
    library = chialvo_map._go_lib if backend == "go" else chialvo_map._mojo_lib
    assert library is not None
    x_output = np.full(3, 17.0, dtype=np.float64)
    y_output = np.full(3, -23.0, dtype=np.float64)
    arguments: list[object] = [0.0, 0.0, 0.89, 0.6, 0.28, 0.04, 1.0, 2, 1.0e308]
    if backend == "go":
        arguments.extend(
            (
                x_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                y_output.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            )
        )
    else:
        arguments.extend((int(x_output.ctypes.data), int(y_output.ctypes.data)))
    result = library.chialvo_map_simulate_complete_c(*arguments)
    assert result == -1
    np.testing.assert_array_equal(x_output, np.full(3, 17.0))
    np.testing.assert_array_equal(y_output, np.full(3, -23.0))


def test_reset_preserves_configuration() -> None:
    neuron = ChialvoMapNeuron(a=0.8, b=0.4, c=0.2, k=0.03, x_threshold=0.75)
    neuron.step(0.01)
    neuron.reset()
    assert (neuron.x, neuron.y) == (0.0, 0.0)
    assert (neuron.a, neuron.b, neuron.c, neuron.k, neuron.x_threshold) == (
        0.8,
        0.4,
        0.2,
        0.03,
        0.75,
    )
