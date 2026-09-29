# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Public Mojo dense IF native dispatch acceptance

"""Exercise actual Mojo ownership through every public ConvertedSNN runtime method."""

import gc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode
from tests.test_accel_mojo_if_abi import library as library
from tests.test_conversion_go_native import assert_bits


@pytest.mark.parametrize("mode", ["spikes", "linear"])
@pytest.mark.parametrize("shape", [(7, 3), (0, 2), (5, 0)])
def test_mojo_complete_continuation_and_retained_slices(
    library: Path, monkeypatch: pytest.MonkeyPatch, mode: OutputMode, shape: tuple[int, int]
) -> None:
    """Require complete public results, chunk continuation and retained-owner independence."""
    monkeypatch.delenv("SC_NEUROCORE_IF_RUST_LIB", raising=False)
    monkeypatch.delenv("SC_NEUROCORE_IF_GO_LIB", raising=False)
    monkeypatch.setenv("SC_NEUROCORE_IF_MOJO_LIB", str(library))
    rng = np.random.default_rng(27)
    model = ConvertedSNN(
        [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))],
        [None, rng.normal(0, 0.1, 2)],
        [0.75, 1.25],
        T=129,
        output_mode=mode,
        layer_membrane_fractions=[0.0, 0.5],
    )
    steps, batch = shape
    frames = rng.random((steps, batch, 3))
    expected = model.replay(frames, trace=True, binary_inputs=False, backend="numpy")
    actual = model.replay(frames, trace=True, binary_inputs=False, backend="mojo")
    assert_bits(actual, expected)
    assert_bits(model.replay(frames, trace=True, binary_inputs=False), expected)
    split = steps // 2
    first = model.replay(frames[:split], binary_inputs=False, backend="mojo")
    second = model.replay(
        frames[split:], initial_state=first.final_state, binary_inputs=False, backend="mojo"
    )
    for left, right in zip(second.final_state, expected.final_state, strict=True):
        assert left.tobytes() == right.tobytes()
    combined = first.output + second.output if mode == "spikes" else second.output
    assert combined.tobytes() == expected.output.tobytes()
    held = actual.final_state[0][:]
    saved = held.copy()
    frames.fill(0)
    model.weights[0].fill(99)
    del actual, first, second
    gc.collect()
    assert held.tobytes() == saved.tobytes()
    held.fill(-99)
    assert expected.final_state[0].tobytes() == saved.tobytes()
    for encoding in ("constant", "poisson"):
        values = rng.random((batch, 3))
        expected_counts = model.run(values, input_mode=encoding, seed=71, backend="numpy")
        assert (
            model.run(values, input_mode=encoding, seed=71, backend="mojo").tobytes()
            == expected_counts.tobytes()
        )
        assert (
            model.rates(values, input_mode=encoding, seed=71, backend="mojo").tobytes()
            == (expected_counts / 129).tobytes()
        )
        np.testing.assert_array_equal(
            model.classify(values, backend="mojo"), model.classify(values, backend="numpy")
        )


def test_mojo_selection_refusals_and_concurrent_owners(
    library: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Admit exact block budgets, reject broken selections and own concurrent results independently."""
    for backend in ("RUST", "GO", "MOJO"):
        monkeypatch.delenv(f"SC_NEUROCORE_IF_{backend}_LIB", raising=False)
    model = ConvertedSNN([[[1.0]]], [None], [1.0], T=129)
    empty = np.empty((0, 1))
    with pytest.raises(RuntimeError, match="SC_NEUROCORE_IF_MOJO_LIB"):
        model.run(empty, backend="mojo")
    monkeypatch.setenv("SC_NEUROCORE_IF_MOJO_LIB", str(tmp_path / "absent.so"))
    with pytest.raises(RuntimeError, match="unavailable"):
        model.run(empty, backend="mojo")
    monkeypatch.setenv("SC_NEUROCORE_IF_MOJO_LIB", str(library))
    assert model.run(empty, backend="mojo").shape == (0, 1)
    assert model.run([1.0], backend="mojo", max_working_bytes=1136).tolist() == [129]
    with pytest.raises(MemoryError):
        model.run([1.0], backend="mojo", max_working_bytes=1135)
    with pytest.raises(ValueError):
        model.replay([[[0.5]]], backend="mojo")
    extreme = np.finfo(np.float64).max
    overflowing = ConvertedSNN([[[extreme]]], [[extreme]], [1.0], T=1)
    with pytest.raises(FloatingPointError):
        overflowing.replay([[[1.0]]], backend="mojo")
    monkeypatch.setenv("SC_NEUROCORE_IF_GO_LIB", str(tmp_path / "absent-go.so"))
    with pytest.raises(RuntimeError, match="unavailable"):
        model.run(empty)
    assert model.run([1.0], backend="mojo").tolist() == [129]
    frames = np.ones((129, 2, 1))
    expected = model.replay(frames, trace=True, backend="numpy")
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: model.replay(frames, trace=True, backend="mojo"), range(16))
        )
    for actual in results:
        assert_bits(actual, expected)
    results[0].final_state[0].fill(-99)
    results[0].state_trace[0].fill(-99)
    for actual in results[1:]:
        assert_bits(actual, expected)
