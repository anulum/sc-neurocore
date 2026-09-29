# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public Go dense IF native dispatch acceptance

"""Exercise pinned Go ownership, concurrent replay and encoded public methods."""

import gc
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode
from sc_neurocore.conversion.if_replay import IFReplayResult


@pytest.fixture(scope="module")
def library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the real Go replay with strict cgo checks and the selected race profile."""
    root = Path(__file__).resolve().parents[1]
    target = tmp_path_factory.mktemp("go-if-public-race") / "if-replay.so"
    flags = ["-race"] if os.environ.get("SC_NEUROCORE_IF_TEST_RACE_PROFILE") == "1" else []
    subprocess.run(
        ["go", "build", *flags, "-buildmode=c-shared", "-o", str(target), "./conversion/cshared"],
        cwd=root / "src/sc_neurocore/accel/go",
        env=dict(os.environ, GOEXPERIMENT="cgocheck2"),
        check=True,
        capture_output=True,
        timeout=120,
    )
    return target


def assert_bits(actual: IFReplayResult, expected: IFReplayResult) -> None:
    """Compare complete shaped output, state and event buffers without numeric tolerance."""
    for left, right in zip(
        [(actual.output,), actual.final_state, actual.state_trace, actual.spike_trace],
        [(expected.output,), expected.final_state, expected.state_trace, expected.spike_trace],
        strict=True,
    ):
        for a, b in zip(left, right, strict=True):
            assert a.shape == b.shape
            assert a.tobytes() == b.tobytes()


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
def test_go_auto_replay_continuation_and_array_lifetime(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: OutputMode,
    shape: tuple[int, int],
) -> None:
    """Continue real Go state and retain sliced arrays after all enclosing results expire."""
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(library))
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
    expected = model.replay(frames, trace=True, binary_inputs=False, backend="numpy")
    actual = model.replay(frames, trace=True, binary_inputs=False, backend="go")
    assert_bits(actual, expected)
    assert_bits(model.replay(frames, trace=True, binary_inputs=False), expected)
    split = steps // 2
    first = model.replay(frames[:split], backend="go", binary_inputs=False)
    second = model.replay(
        frames[split:], initial_state=first.final_state, backend="go", binary_inputs=False
    )
    for a, b in zip(second.final_state, expected.final_state, strict=True):
        assert a.tobytes() == b.tobytes()
    held = actual.final_state[0][:]
    saved = held.copy()
    frames.fill(0)
    model.weights[0].fill(99)
    del actual, first, second
    gc.collect()
    assert held.tobytes() == saved.tobytes()
    held.fill(-99)
    assert expected.final_state[0].tobytes() == saved.tobytes()


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("encoding", ["poisson", "constant"])
def test_go_encoded_blocks_rates_and_first_maximum(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: OutputMode,
    encoding: Literal["poisson", "constant"],
) -> None:
    """Exercise every public method across three native blocks with the same seeded encoder."""
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(library))
    model = ConvertedSNN(
        [[[0.75, -0.25], [0.25, 0.5]]],
        [[0.1, -0.2]],
        [0.75],
        T=129,
        output_mode=mode,
        output_scale=3.0,
    )
    for values in (np.array([0.75, 0.25]), np.array([[0.75, 0.25], [0.1, 0.9]])):
        expected = model.run(values, input_mode=encoding, seed=71, backend="numpy")
        actual = model.run(values, input_mode=encoding, seed=71, backend="go")
        assert actual.tobytes() == expected.tobytes()
        assert (
            model.rates(values, input_mode=encoding, seed=71, backend="go").tobytes()
            == (expected / 129 * 3).tobytes()
        )
        np.testing.assert_array_equal(
            model.classify(values, backend="go"), model.classify(values, backend="numpy")
        )


def test_go_selection_and_numeric_refusals_preserve_floor(
    library: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Refuse absent Go, broken preferred Rust, and exact native memory overruns through the API."""
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_IF_GO_LIB", raising=False)
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
    empty = np.empty((0, 1))
    with pytest.raises(RuntimeError, match="SC_NEUROCORE_IF_GO_LIB"):
        model.run(empty, backend="go")
    monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(library))
    assert model.run(empty, backend="go").shape == (0, 1)
    assert model.run([1.0], backend="go", max_working_bytes=1136).tolist() == [129]
    with pytest.raises(MemoryError):
        model.run([1.0], backend="go", max_working_bytes=1135)
    with pytest.raises(ValueError):
        model.replay([[[0.5]]], backend="go")
    extreme = np.finfo(np.float64).max
    overflowing = ConvertedSNN([[[extreme]]], [[extreme]], [1.0], T=1)
    with pytest.raises(FloatingPointError):
        overflowing.replay([[[1.0]]], backend="go")
    monkeypatch.setenv("SC_NEUROCORE_IF_RUST_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="unavailable"):
        model.run(empty)
    assert model.run([1.0], backend="go").tolist() == [129]
    assert model.run([1.0], backend="numpy").tolist() == [129]


def test_go_pinned_parallel_results_are_independent(
    library: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay concurrently into independently owned pinned result buffers."""
    monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(library))
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
    frames = np.ones((129, 2, 1))
    expected = model.replay(frames, trace=True, backend="numpy")

    def replay(_: int) -> IFReplayResult:
        return model.replay(frames, trace=True, backend="go")

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(replay, range(16)))
    for actual in results:
        assert_bits(actual, expected)
    results[0].final_state[0].fill(-99)
    results[0].state_trace[0].fill(-99)
    for actual in results[1:]:
        assert_bits(actual, expected)


def test_go_public_native_race_profile(tmp_path: Path) -> None:
    """Run the same public API corpus with native race detection in a compatible loader process."""
    root = Path(__file__).resolve().parents[1]
    environment = dict(
        os.environ,
        LD_PREFER_MAP_32BIT_EXEC="1",
        SC_NEUROCORE_IF_TEST_RACE_PROFILE="1",
        GORACE=f"log_path={tmp_path / 'race-report'}",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-s",
            "-q",
            str(Path(__file__).resolve()),
            "-k",
            "not test_go_public_native_race_profile",
            "--basetemp",
            str(tmp_path / "native-corpus"),
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    reports = list(tmp_path.glob("race-report.*"))
    details = "\n".join(p.read_text() for p in reports)
    assert result.returncode == 0, result.stdout + result.stderr + details
    assert "12 passed" in result.stdout
    assert not reports, details
