# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public dense IF native runtime acceptance

"""Exercise compiled Rust dispatch, complete trajectories and array lifetime through the API."""

import gc
import subprocess
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode
from sc_neurocore.conversion.if_replay import IFReplayResult
from sc_neurocore.conversion.if_dispatch import ReplayBackend


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the maintained ownership ABI for actual public API selection."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("rust-if-runtime")
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
    return target / "release/libsc_neurocore_if_replay.so"


def assert_bits(actual: IFReplayResult, expected: IFReplayResult) -> None:
    """Compare complete numerical responses and each requested state/event trajectory."""
    for left, right in zip(
        [(actual.output,), actual.final_state, actual.state_trace, actual.spike_trace],
        [(expected.output,), expected.final_state, expected.state_trace, expected.spike_trace],
        strict=True,
    ):
        assert len(left) == len(right)
        for a, b in zip(left, right, strict=True):
            assert a.shape == b.shape
            assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
@pytest.mark.parametrize("trace", [False, True])
def test_public_native_full_trajectories_and_continuation(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: OutputMode,
    shape: tuple[int, int],
    trace: bool,
) -> None:
    """Select actual Rust/auto/NumPy and continue owned states across two real blocks."""
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(library))
    steps, batch = shape
    rng = np.random.default_rng(27)
    model = ConvertedSNN(
        [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))],
        [None, rng.normal(0, 0.1, 2)],
        [0.75, 1.25],
        T=129,
        output_mode=mode,
        layer_membrane_fractions=[0.0, 0.5],
    )
    frames = rng.random((steps, batch, 3))
    expected = model.replay(frames, trace=trace, binary_inputs=False, backend="numpy")
    actual = model.replay(frames, trace=trace, binary_inputs=False, backend="rust")
    assert_bits(actual, expected)
    assert_bits(model.replay(frames, trace=trace, binary_inputs=False), expected)
    split = steps // 2
    first = model.replay(frames[:split], binary_inputs=False, backend="rust")
    second = model.replay(
        frames[split:],
        initial_state=first.final_state,
        trace=trace,
        binary_inputs=False,
        backend="rust",
    )
    floor_first = model.replay(frames[:split], binary_inputs=False, backend="numpy")
    floor_second = model.replay(
        frames[split:],
        initial_state=floor_first.final_state,
        trace=trace,
        binary_inputs=False,
        backend="numpy",
    )
    assert_bits(second, floor_second)
    held = actual.final_state[0][:]
    saved = held.copy()
    frames.fill(0)
    model.weights[0].fill(100)
    del actual, first, second
    gc.collect()
    np.testing.assert_array_equal(held, saved)
    held.fill(-12)
    np.testing.assert_array_equal(expected.final_state[0], saved)


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("input_mode", ["poisson", "constant"])
@pytest.mark.parametrize("vector", [False, True])
def test_public_encoded_runtime_rates_and_classification(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: OutputMode,
    input_mode: Literal["poisson", "constant"],
    vector: bool,
) -> None:
    """Preserve seeded encoding and native state across three blocks through every public method."""
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(library))
    model = ConvertedSNN(
        [[[0.75, -0.25], [0.25, 0.5]]],
        [[0.1, -0.2]],
        [0.75],
        T=129,
        output_mode=mode,
        output_scale=3.0,
    )
    values = np.array([0.75, 0.25]) if vector else np.array([[0.75, 0.25], [0.1, 0.9]])
    expected = model.run(values, input_mode=input_mode, seed=71, backend="numpy")
    for backend in ("rust", "auto"):
        actual = model.run(values, input_mode=input_mode, seed=71, backend=backend)
        assert actual.tobytes() == expected.tobytes()
        rates = model.rates(values, input_mode=input_mode, seed=71, backend=backend)
        assert rates.tobytes() == (expected / 129 * 3).tobytes()
        np.testing.assert_array_equal(
            model.classify(values, backend=backend), model.classify(values, backend="numpy")
        )


def test_backend_refusals_empty_batches_and_floor_override(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refuse missing/invalid native selection even without encoded timesteps, preserving the floor."""
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
    empty = np.empty((0, 1))
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    assert model.run(empty).shape == (0, 1)
    with pytest.raises(RuntimeError, match="requires"):
        model.run(empty, backend="rust")
    with pytest.raises(ValueError, match="unsupported"):
        model.run(empty, backend=cast(ReplayBackend, "unknown"))
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="unavailable"):
        model.run(empty)
    assert model.run([1.0], backend="numpy").tolist() == [129]
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(library))
    assert model.run(empty, backend="rust").shape == (0, 1)
    with pytest.raises(MemoryError):
        model.replay([[[1.0]]], trace=True, max_working_bytes=127, backend="rust")
    with pytest.raises(ValueError):
        model.replay([[[0.5]]], backend="rust")
    extreme = np.finfo(np.float64).max
    overflowing = ConvertedSNN([[[extreme]]], [[extreme]], [1.0], T=1)
    with pytest.raises(FloatingPointError, match="overflow"):
        overflowing.replay([[[1.0]]], backend="rust")


def test_encoded_native_reserves_previous_owner_before_execution(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Account for the previous opaque result while admitting the next native block."""
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(library))
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
    # A 64-frame block reserves 1120 bytes; its retained state/output add 16.
    with pytest.raises(MemoryError):
        model.run([1.0], backend="rust", max_working_bytes=1135)
    assert model.run([1.0], backend="rust", max_working_bytes=1136).tolist() == [129]
    with pytest.raises(MemoryError):
        model.run(np.ones((100, 1)), backend="rust", max_working_bytes=16)
