# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Go complete dense IF replay parity

"""Compare real Go output/state/event vectors with the public Python runtime."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


def _go_values(values: npt.NDArray[np.float64]) -> str:
    """Encode canonical row-major float64 literals for the native caller."""
    return "[]float64{" + ",".join(repr(float(v)) for v in values.ravel()) + "}"


def test_go_native_complete_state_and_chunk_continuation(tmp_path: Path) -> None:
    """Require exact bit parity for mixed preloads, signed output and empty geometry."""
    root = Path(__file__).resolve().parents[1]
    module = root / "src/sc_neurocore/accel/go"
    program = [
        'package main\nimport ("encoding/binary"; "math"; "os"; "reflect"; c "github.com/anulum/sc-neurocore/accel/conversion")',
        "func main() {",
    ]
    expected: list[bytes] = []
    for mode in ("spikes", "linear"):
        output_mode: OutputMode = "linear" if mode == "linear" else "spikes"
        for binary in (False, True):
            for steps, batch in ((7, 3), (0, 2), (5, 0)):
                for seed in range(4):
                    rng = np.random.default_rng(seed)
                    weights = [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))]
                    biases = [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)]
                    network = ConvertedSNN(
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
                        [rng.normal(0.0, 0.5, (batch, 4)), rng.normal(0.0, 0.5, (batch, 2))]
                        if seed % 2
                        else None
                    )
                    result = network.replay(
                        frames, initial_state=state, trace=True, binary_inputs=binary
                    )
                    for values in (
                        result.output,
                        *result.final_state,
                        *result.state_trace,
                        *result.spike_trace,
                    ):
                        expected.append(values.astype("<f8").tobytes())
                    program.append("{")
                    for i, weight in enumerate(weights):
                        bias = biases[i]
                        program.append(
                            f"layer{i},err:=c.NewDenseLayer({weight.shape[1]},{weight.shape[0]},"
                            f"{_go_values(weight)},{'nil' if bias is None else _go_values(bias)},"
                            f"{[0.75, 1.25][i]},{[0.0, 0.5][i]},1<<20);if err!=nil {{panic(err)}}"
                        )
                    program.append(
                        f'model,err:=c.NewConvertedSNN([]c.DenseLayer{{layer0,layer1}},c.OutputMode("{mode}"),1<<20);if err!=nil {{panic(err)}}'
                    )
                    initial = (
                        "nil"
                        if state is None
                        else "[][]float64{" + ",".join(_go_values(v) for v in state) + "}"
                    )
                    program.append(f"frames:={_go_values(frames)}")
                    program.append(
                        f"options:=c.ReplayOptions{{InitialState:{initial},Trace:true,"
                        f"BinaryInputs:{str(binary).lower()},MaxWorkingBytes:1<<20}}"
                    )
                    program.append(
                        f"result,err:=model.Replay(frames,{steps},{batch},options);if err!=nil {{panic(err)}}"
                    )
                    program.append(
                        "vectors:=[][]float64{result.Output};vectors=append(vectors,result.FinalState...);"
                        "vectors=append(vectors,result.StateTrace...);vectors=append(vectors,result.SpikeTrace...);"
                        "for _,values:=range vectors {for _,value:=range values {"
                        "if err:=binary.Write(os.Stdout,binary.LittleEndian,math.Float64bits(value));err!=nil {panic(err)}}}"
                    )
                    labels = ",".join(str(int(v)) for v in np.argmax(result.output, axis=-1))
                    program.append(
                        f'labels,err:=model.Classify(result,{batch});if err!=nil || !reflect.DeepEqual(labels,[]int{{{labels}}}) {{panic("classification mismatch")}}'
                    )
                    if steps and batch:
                        offset = 3 * batch * 3
                        program.append(
                            f"first,err:=model.Replay(frames[:{offset}],3,{batch},options);if err!=nil {{panic(err)}}"
                        )
                        program.append("options.InitialState=first.FinalState")
                        program.append(
                            f"second,err:=model.Replay(frames[{offset}:],{steps - 3},{batch},options);if err!=nil {{panic(err)}}"
                        )
                        if mode == "spikes":
                            program.append(
                                "for i:=range second.Output {second.Output[i]+=first.Output[i]}"
                            )
                        program.append(
                            'if !reflect.DeepEqual(second.Output,result.Output) || !reflect.DeepEqual(second.FinalState,result.FinalState) {panic("continuation mismatch")}'
                        )
                        for name in ("StateTrace", "SpikeTrace"):
                            program.append(
                                f'for i:=range result.{name} {{joined:=append(append([]float64(nil),first.{name}[i]...),second.{name}[i]...);if !reflect.DeepEqual(joined,result.{name}[i]) {{panic("trace continuation mismatch")}}}}'
                            )
                    program.append("}")
    program.append("}")
    caller = tmp_path / "complete_replay.go"
    caller.write_text("\n".join(program))
    completed = subprocess.run(
        ["go", "run", str(caller)],
        cwd=module,
        capture_output=True,
        check=True,
        timeout=120,
    )
    assert completed.stdout == b"".join(expected)
