# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo complete dense IF replay parity

"""Compile and exercise real Mojo replay against the public Python reference."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


def _values(values: npt.NDArray[np.float64]) -> str:
    """Serialize canonical row-major doubles into a typed native List literal."""
    return "[" + ",".join(repr(float(v)) for v in values.ravel()) + "]"


def test_mojo_complete_state_and_chunk_continuation(tmp_path: Path) -> None:
    """Require all output/state/event bits and native continuation to match Python."""
    root = Path(__file__).resolve().parents[1]
    kernels = root / "src/sc_neurocore/accel/mojo/kernels"
    source = [
        r'''"""Exercise complete native replay results through the actual Mojo API."""
from std.ffi import external_call
from ann_to_snn import replay, classify
from ann_to_snn_parameters import DenseLayer, ConvertedSNN


def emit(values: List[Float64]) raises:
    """Write host little-endian result doubles to the parity receipt stream.

    Args:
        values: Actual exported replay response or state/event buffer.

    Raises:
        Error: Incomplete native stdout transport.
    """
    var size = len(values) * 8
    var offset = 0
    while offset < size:
        var count = external_call["write", Int](Int(1), values.unsafe_ptr().unsafe_bitcast[UInt8]().unsafe_offset(offset), size-offset)
        if count <= 0:
            raise Error("incomplete parity receipt")
        offset += count


def check(a: List[Float64], b: List[Float64]) raises:
    """Require actual native continuation to reproduce the whole native replay.

    Args:
        a: First actual native replay vector.
        b: Second actual native replay vector.

    Raises:
        Error: Dimension or response mismatch.
    """
    if len(a) != len(b):
        raise Error("continuation shape mismatch")
    for i in range(len(a)):
        if a[i] != b[i]:
            raise Error("continuation response mismatch")
'''
    ]
    expected: list[bytes] = []
    index = 0
    for mode in ("spikes", "linear"):
        output_mode: OutputMode = "linear" if mode == "linear" else "spikes"
        for binary in (False, True):
            for steps, batch in ((7, 3), (0, 2), (5, 0)):
                for seed in range(4):
                    rng = np.random.default_rng(seed)
                    weights = [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))]
                    biases = [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)]
                    model = ConvertedSNN(
                        weights,
                        biases,
                        [0.75, 1.25],
                        T=7,
                        output_mode=output_mode,
                        layer_membrane_fractions=[0.0, 0.5],
                    )
                    frames = rng.random((steps, batch, 3))
                    if binary:
                        frames = (frames < 0.5).astype(np.float64)
                    state = (
                        [rng.normal(0, 0.5, (batch, 4)), rng.normal(0, 0.5, (batch, 2))]
                        if seed % 2
                        else None
                    )
                    result = model.replay(
                        frames, initial_state=state, trace=True, binary_inputs=binary
                    )
                    for values in (
                        result.output,
                        *result.final_state,
                        *result.state_trace,
                        *result.spike_trace,
                    ):
                        expected.append(values.astype("<f8").tobytes())
                    source.append(f'''\ndef case{index}() raises:
    """Exercise complete native case {index}.

    Raises:
        Error: Native admission, arithmetic, transport or continuation mismatch.
    """
    var layers = List[DenseLayer]()''')
                    for i, weight in enumerate(weights):
                        bias = biases[i]
                        source.append(f"    var weight{i}: List[Float64] = {_values(weight)}")
                        source.append(
                            f"    var bias{i}: List[Float64] = {'[]' if bias is None else _values(bias)}"
                        )
                        source.append(
                            f"    layers.append(DenseLayer({weight.shape[1]},{weight.shape[0]},weight{i},bias{i},{[0.75, 1.25][i]},{[0.0, 0.5][i]}))"
                        )
                    source.append(f"    var model = ConvertedSNN(layers,linear={mode == 'linear'})")
                    source.append(f"    var frames: List[Float64] = {_values(frames)}")
                    initial = (
                        "[]" if state is None else "[" + ",".join(_values(v) for v in state) + "]"
                    )
                    source.append(f"    var initial: List[List[Float64]] = {initial}")
                    source.append(
                        f"    var result = replay(model,frames,{steps},{batch},initial_state=initial,use_initial_state={state is not None},trace=True,binary_inputs={binary})"
                    )
                    source.append("    emit(result.output)")
                    for name in ("final_state", "state_trace", "spike_trace"):
                        source.append(
                            f"    for i in range(len(result.{name})):\n        emit(result.{name}[i])"
                        )
                    source.append(f"    var labels = classify(model,result,{batch})")
                    for row, label in enumerate(np.argmax(result.output, axis=-1)):
                        source.append(
                            f'    if labels[{row}] != {int(label)}:\n        raise Error("classification mismatch")'
                        )
                    if steps and batch:
                        offset = 3 * batch * 3
                        source.append(
                            "    var prefix = List[Float64]()\n    var suffix = List[Float64]()"
                        )
                        source.append(
                            f"    for i in range({offset}):\n        prefix.append(frames[i])"
                        )
                        source.append(
                            f"    for i in range({offset},len(frames)):\n        suffix.append(frames[i])"
                        )
                        source.append(
                            f"    var first = replay(model,prefix,3,{batch},initial_state=initial,use_initial_state={state is not None},trace=True,binary_inputs={binary})"
                        )
                        source.append(
                            f"    var second = replay(model,suffix,{steps - 3},{batch},initial_state=first.final_state,use_initial_state=True,trace=True,binary_inputs={binary})"
                        )
                        if mode == "spikes":
                            source.append(
                                "    for i in range(len(second.output)):\n        second.output[i] += first.output[i]"
                            )
                        source.append(
                            "    check(second.output,result.output)\n    for i in range(len(result.final_state)):\n        check(second.final_state[i],result.final_state[i])"
                        )
                        for name in ("state_trace", "spike_trace"):
                            source.append(
                                f"    for i in range(len(result.{name})):\n        var joined = first.{name}[i].copy()\n        for value in second.{name}[i]:\n            joined.append(value)\n        check(joined,result.{name}[i])"
                            )
                    index += 1
    source.append('''\ndef main() raises:
    """Run every complete native case through the compiled production modules.

    Raises:
        Error: Native replay refusal or parity mismatch.
    """''')
    for i in range(index):
        source.append(f"    case{i}()")
    caller = tmp_path / "complete_replay.mojo"
    caller.write_text("\n".join(source))
    executable = tmp_path / "complete_replay"
    subprocess.run(
        [
            "mojo",
            "build",
            str(caller),
            "-I",
            str(kernels),
            "--Werror",
            "--diagnose-missing-doc-strings",
            "--fp-mode",
            "contract=off",
            "-j",
            "2",
            "-o",
            str(executable),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    completed = subprocess.run([str(executable)], capture_output=True, check=True, timeout=30)
    assert completed.stdout == b"".join(expected)
