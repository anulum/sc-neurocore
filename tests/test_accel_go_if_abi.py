# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go pinned dense IF C ownership acceptance

"""Exercise the real Go C request and pinned-result ownership ABI with strict cgo checks."""

import ctypes
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_native import load_native
from sc_neurocore.conversion.if_native_types import BufferView, LayerSpec, ReplayRequest
from sc_neurocore.conversion.if_parameters import OutputMode


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Compile the maintained Go replay ABI with the full cgo pointer-check experiment."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("go-if-abi") / "if-replay.so"
    environment = dict(os.environ, GOEXPERIMENT="cgocheck2")
    subprocess.run(
        ["go", "build", "-buildmode=c-shared", "-o", str(target), "./conversion/cshared"],
        cwd=root / "src/sc_neurocore/accel/go",
        env=environment,
        capture_output=True,
        check=True,
        timeout=120,
    )
    return target


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
def test_pinned_complete_result_buffers_and_caller_custody(
    library: Path,
    mode: OutputMode,
    binary: bool,
    shape: tuple[int, int],
) -> None:
    """Call exported Go C symbols directly and compare every buffer with public NumPy replay."""
    api = load_native(str(library))
    steps, batch = shape
    for seed in range(4):
        rng = np.random.default_rng(seed)
        weights = [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))]
        biases = [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)]
        frames = rng.random((steps, batch, 3))
        if binary:
            frames = (frames < 0.5).astype(np.float64)
        initial = (
            [rng.normal(0, 0.5, (batch, 4)), rng.normal(0, 0.5, (batch, 2))] if seed % 2 else None
        )
        model = ConvertedSNN(
            weights,
            biases,
            [0.75, 1.25],
            T=7,
            output_mode=mode,
            layer_membrane_fractions=[0.0, 0.5],
        )
        expected = model.replay(
            frames, initial_state=initial, trace=True, binary_inputs=binary, backend="numpy"
        )
        specs = (LayerSpec * 2)()
        for i in range(2):
            bias, state = biases[i], None if initial is None else initial[i]
            specs[i] = LayerSpec(
                weights[i].shape[0],
                weights[i].shape[1],
                weights[i].ctypes.data,
                0 if bias is None else bias.ctypes.data,
                0 if bias is None else bias.size,
                [0.75, 1.25][i],
                [0.0, 0.5][i],
                0 if state is None else state.ctypes.data,
                0 if state is None else state.size,
            )
        flags = (
            1
            | (2 if binary else 0)
            | (4 if mode == "linear" else 0)
            | (8 if initial is not None else 0)
        )
        request = ReplayRequest(
            1,
            flags,
            ctypes.addressof(specs),
            2,
            frames.ctypes.data,
            frames.size,
            steps,
            batch,
            1 << 20,
        )
        handle = ctypes.c_void_p()
        assert api.replay(ctypes.byref(request), ctypes.byref(handle)) == 0
        assert handle.value is not None
        try:
            frames.fill(0)
            weights[0].fill(99)
            if initial is not None:
                initial[0].fill(99)
            for kind, arrays in enumerate(
                [
                    (expected.output,),
                    expected.final_state,
                    expected.state_trace,
                    expected.spike_trace,
                ]
            ):
                for index, array in enumerate(arrays):
                    view = BufferView()
                    assert api.buffer(handle, kind, index, ctypes.byref(view)) == 0
                    assert view.length == array.size
                    actual = (
                        (ctypes.c_double * view.length).from_address(view.data)
                        if view.length
                        else (ctypes.c_double * 0)()
                    )
                    assert np.ctypeslib.as_array(actual).tobytes() == array.tobytes()
            for kind, index in [(0, 1), (1, 2), (2, 2), (3, 2), (99, 0)]:
                untouched = BufferView(16, 999)
                assert api.buffer(handle, kind, index, ctypes.byref(untouched)) == -1
                assert (untouched.data, untouched.length) == (16, 999)
            untouched = BufferView(16, 999)
            assert api.buffer(None, 0, 0, ctypes.byref(untouched)) == -1
            assert api.buffer(handle, 0, 0, None) == -1
        finally:
            api.free(handle)


def test_go_categorical_refusal_preserves_owner_slot(library: Path) -> None:
    """Refuse raw request errors, exact numeric limits and finite overflow without a result."""
    api = load_native(str(library))
    weight, frame = np.array([1.0]), np.array([1.0])
    specs = (LayerSpec * 1)(LayerSpec(1, 1, weight.ctypes.data, 0, 0, 1.0, 0.0, 0, 0))
    request = ReplayRequest(1, 3, ctypes.addressof(specs), 1, frame.ctypes.data, 1, 1, 1, 128)
    for field, value, status in [
        ("version", 2, -1),
        ("flags", 32, -1),
        ("max_working_bytes", 127, -2),
        ("max_working_bytes", 0, -1),
        ("layer_count", 1 << 53, -2),
        ("frames", 1, -1),
        ("layers", 0, -1),
    ]:
        original = getattr(request, field)
        setattr(request, field, value)
        handle = ctypes.c_void_p(0x1234)
        assert api.replay(ctypes.byref(request), ctypes.byref(handle)) == status
        assert handle.value == 0x1234
        setattr(request, field, original)
    for frame_value in (float("nan"), 1.5, 0.5):
        frame[0] = frame_value
        handle = ctypes.c_void_p(0x1234)
        assert api.replay(ctypes.byref(request), ctypes.byref(handle)) == -1
        assert handle.value == 0x1234
    frame[0] = 1.0
    weight[0] = np.finfo(np.float64).max
    bias = np.array([np.finfo(np.float64).max])
    specs[0].bias, specs[0].bias_len = bias.ctypes.data, 1
    request.max_working_bytes = 1024
    handle = ctypes.c_void_p(0x1234)
    assert api.replay(ctypes.byref(request), ctypes.byref(handle)) == -3
    assert handle.value == 0x1234
    assert api.replay(None, ctypes.byref(handle)) == -1
    assert api.replay(ctypes.byref(request), None) == -1
    api.free(None)
