# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Rust owned dense IF C boundary acceptance

"""Execute the real Rust C ownership boundary against public Python replay."""

import ctypes
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


class LayerSpec(ctypes.Structure):
    """Native ABI-one borrowed dense layer and optional initial-state descriptor."""

    _fields_ = [
        ("outputs", ctypes.c_size_t),
        ("inputs", ctypes.c_size_t),
        ("weights", ctypes.c_void_p),
        ("bias", ctypes.c_void_p),
        ("bias_len", ctypes.c_size_t),
        ("threshold", ctypes.c_double),
        ("initial_fraction", ctypes.c_double),
        ("initial", ctypes.c_void_p),
        ("initial_len", ctypes.c_size_t),
    ]


class ReplayRequest(ctypes.Structure):
    """ABI-one request with retained caller-owned descriptors and arrays."""

    _fields_ = [
        ("version", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("layers", ctypes.c_void_p),
        ("layer_count", ctypes.c_size_t),
        ("frames", ctypes.c_void_p),
        ("frames_len", ctypes.c_size_t),
        ("steps", ctypes.c_size_t),
        ("batch", ctypes.c_size_t),
        ("max_working_bytes", ctypes.c_size_t),
    ]


class BufferView(ctypes.Structure):
    """Borrowed row-major result storage, live until its opaque owner is freed."""

    _fields_ = [("data", ctypes.c_void_p), ("length", ctypes.c_size_t)]


ReplayCall = Callable[[object, object], int]
BufferCall = Callable[[object, int, int, object], int]
FreeCall = Callable[[object], None]


@pytest.fixture(scope="module")
def native(tmp_path_factory: pytest.TempPathFactory) -> tuple[ReplayCall, BufferCall, FreeCall]:
    """Build and load the real maintained native kernel with its exact C layouts."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("rust-if-abi")
    subprocess.run(
        [
            "cargo",
            "build",
            "--offline",
            "--release",
            "--manifest-path",
            str(root / "src/sc_neurocore/accel/rust/safety/if_native/Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(target / "release/libsc_neurocore_if_replay.so"))
    library.sc_if_abi_version.argtypes = []
    library.sc_if_abi_version.restype = ctypes.c_uint32
    assert library.sc_if_abi_version() == 1
    library.sc_if_replay.argtypes = [ctypes.POINTER(ReplayRequest), ctypes.POINTER(ctypes.c_void_p)]
    library.sc_if_replay.restype = ctypes.c_int32
    library.sc_if_buffer.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_size_t,
        ctypes.POINTER(BufferView),
    ]
    library.sc_if_buffer.restype = ctypes.c_int32
    library.sc_if_free.argtypes = [ctypes.c_void_p]
    library.sc_if_free.restype = None
    return (
        cast(ReplayCall, library.sc_if_replay),
        cast(BufferCall, library.sc_if_buffer),
        cast(FreeCall, library.sc_if_free),
    )


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
def test_opaque_result_complete_bits_and_independent_storage(
    native: tuple[ReplayCall, BufferCall, FreeCall],
    mode: OutputMode,
    binary: bool,
    shape: tuple[int, int],
) -> None:
    """Borrow every real result buffer, then mutate callers without altering owned state."""
    replay, buffer, free = native
    steps, batch = shape
    for seed in range(4):
        rng = np.random.default_rng(seed)
        weights = [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))]
        biases = [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)]
        model = ConvertedSNN(
            weights,
            biases,
            [0.75, 1.25],
            T=7,
            output_mode=mode,
            layer_membrane_fractions=[0.0, 0.5],
        )
        frames = rng.random((steps, batch, 3))
        if binary:
            frames = (frames < 0.5).astype(np.float64)
        initial = (
            [rng.normal(0, 0.5, (batch, 4)), rng.normal(0, 0.5, (batch, 2))] if seed % 2 else None
        )
        expected = model.replay(frames, initial_state=initial, trace=True, binary_inputs=binary)
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
        assert replay(ctypes.byref(request), ctypes.byref(handle)) == 0
        assert handle.value is not None
        try:
            frames.fill(0.75)
            weights[0].fill(100)
            if initial is not None:
                initial[0].fill(100)
            groups = [
                (expected.output,),
                expected.final_state,
                expected.state_trace,
                expected.spike_trace,
            ]
            borrowed: list[npt.NDArray[np.float64]] = []
            for kind, vectors in enumerate(groups):
                for index, values in enumerate(vectors):
                    view = BufferView()
                    assert buffer(handle, kind, index, ctypes.byref(view)) == 0
                    assert view.length == values.size
                    array = np.ctypeslib.as_array(
                        (ctypes.c_double * view.length).from_address(view.data)
                    )
                    assert array.tobytes() == values.tobytes()
                    borrowed.append(array)
            untouched = BufferView(16, 999)
            assert buffer(handle, 99, 0, ctypes.byref(untouched)) == -1
            assert (untouched.data, untouched.length) == (16, 999)
            if expected.final_state[0].size:
                old_trace = borrowed[3].copy()
                borrowed[1].fill(-100)
                np.testing.assert_array_equal(borrowed[3], old_trace)
        finally:
            free(handle)


def test_refusal_leaves_owner_slot_unchanged(
    native: tuple[ReplayCall, BufferCall, FreeCall],
) -> None:
    """Reject invalid layouts, exact budget overruns and overflows without publishing state."""
    replay, _, free = native
    weight = np.array([1.0])
    frame = np.array([1.0])
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
        owner = ctypes.c_void_p(0x1234)
        assert replay(ctypes.byref(request), ctypes.byref(owner)) == status
        assert owner.value == 0x1234
        setattr(request, field, original)
    for frame_value in (float("nan"), 1.5, 0.5):
        frame[0] = frame_value
        owner = ctypes.c_void_p(0x1234)
        assert replay(ctypes.byref(request), ctypes.byref(owner)) == -1
        assert owner.value == 0x1234
    frame[0] = 1.0
    weight[0] = np.finfo(np.float64).max
    bias = np.array([np.finfo(np.float64).max])
    specs[0].bias = bias.ctypes.data
    specs[0].bias_len = 1
    request.max_working_bytes = 1024
    owner = ctypes.c_void_p(0x1234)
    assert replay(ctypes.byref(request), ctypes.byref(owner)) == -3
    assert owner.value == 0x1234
    assert replay(None, ctypes.byref(owner)) == -1
    assert replay(ctypes.byref(request), None) == -1
    free(None)
