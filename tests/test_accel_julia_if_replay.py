# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia complete dense IF replay parity

"""Exercise Julia's native public replay with Python output/state/event bit receipts."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode
from tests.julia_runtimes import require_julia_runtime


def _julia_values(values: npt.NDArray[np.float64]) -> str:
    """Serialize real coefficient or frame vectors in canonical row-major order."""
    return "Float64[" + ",".join(repr(float(v)) for v in values.ravel()) + "]"


@pytest.mark.parametrize("runtime", ["1.11", "1.13"])
def test_native_julia_complete_replay_parity_and_admission(tmp_path: Path, runtime: str) -> None:
    """Match every emitted bit across mixed preload, continuation, signed and empty replay."""
    binaries = [require_julia_runtime(runtime)]
    assert binaries, f"Julia {runtime} native runtime required"
    api = (
        Path(__file__).resolve().parents[1]
        / "src/sc_neurocore/accel/julia/conversion/ann_to_snn.jl"
    )
    program = ["using Test", "include(ARGS[1])", "using .AnnToSnnAccel"]
    expected: list[bytes] = []
    cases = 0
    for mode in ("spikes", "linear"):
        output_mode: OutputMode = "linear" if mode == "linear" else "spikes"
        for binary in (False, True):
            for steps, batch in ((7, 3), (0, 2), (5, 0)):
                for seed in range(4):
                    rng = np.random.default_rng(seed)
                    weights = [rng.normal(0.3, 0.4, (4, 3)), rng.normal(0.1, 0.5, (2, 4))]
                    biases = [None if seed % 2 else rng.normal(0, 0.1, 4), rng.normal(0, 0.1, 2)]
                    thresholds = [0.75, 1.25]
                    fractions = [0.0, 0.5]
                    network = ConvertedSNN(
                        weights,
                        biases,
                        thresholds,
                        T=7,
                        output_mode=output_mode,
                        layer_membrane_fractions=fractions,
                    )
                    frames = rng.random((steps, batch, 3))
                    if binary:
                        frames = (frames < 0.5).astype(np.float64)
                    initial = [rng.normal(0.0, 0.5, (batch, 4)), rng.normal(0.0, 0.5, (batch, 2))]
                    state = initial if seed % 2 else None
                    result = network.replay(
                        frames, initial_state=state, trace=True, binary_inputs=binary
                    )
                    constructors = []
                    for i, weight in enumerate(weights):
                        bias = biases[i]
                        constructors.append(
                            f"DenseLayer({weight.shape[1]},{weight.shape[0]},{_julia_values(weight)},"
                            f"{'nothing' if bias is None else _julia_values(bias)},"
                            f"{thresholds[i]};initial_fraction={fractions[i]})"
                        )
                    program.append("let")
                    program.append(
                        f"model=ConvertedSNN([{','.join(constructors)}];output_mode=:{mode})"
                    )
                    original = (
                        "nothing"
                        if state is None
                        else "[" + ",".join(_julia_values(v) for v in state) + "]"
                    )
                    program.append(f"frames={_julia_values(frames)}")
                    program.append(
                        f"result=@inferred replay(model,frames,({steps},{batch});"
                        f"initial_state={original},trace=true,binary_inputs={str(binary).lower()})"
                    )
                    program.append(
                        "for values in (result.output, result.final_state..., result.state_trace..., result.spike_trace...)\n"
                        "for value in values; write(stdout,htol(reinterpret(UInt64,value))); end\nend"
                    )
                    for values in (
                        result.output,
                        *result.final_state,
                        *result.state_trace,
                        *result.spike_trace,
                    ):
                        expected.append(values.astype("<f8").tobytes())
                    program.append(
                        f"@test classify(model,result,{batch}) == {np.argmax(result.output, axis=-1).tolist()}"
                    )
                    if steps and batch:
                        offset = 3 * batch * 3
                        program.append(
                            f"first=replay(model,frames[1:{offset}],(3,{batch});"
                            f"initial_state={original},trace=true,binary_inputs={str(binary).lower()})"
                        )
                        program.append(
                            f"second=replay(model,frames[{offset + 1}:end],({steps - 3},{batch});"
                            f"initial_state=first.final_state,trace=true,binary_inputs={str(binary).lower()})"
                        )
                        program.append(
                            "@test second.output == result.output"
                            if mode == "linear"
                            else "@test first.output + second.output == result.output"
                        )
                        program.append("@test second.final_state == result.final_state")
                        program.append(
                            "@test [vcat(a,b) for (a,b) in zip(first.state_trace,second.state_trace)] == result.state_trace"
                        )
                        program.append(
                            "@test [vcat(a,b) for (a,b) in zip(first.spike_trace,second.spike_trace)] == result.spike_trace"
                        )
                    program.append("end")
                    cases += 1
    program.append(r"""
let
    weights=[1.0]; bias=[0.25]
    layer=DenseLayer(1,1,weights,bias,1.0; initial_fraction=0.5)
    model=ConvertedSNN([layer])
    weights[1]=100.0; bias[1]=100.0; layer.weights[1]=100.0
    result=@inferred replay(model,[0.5],(1,1);trace=true,binary_inputs=false)
    @test result.output == [1.0]
    @test result.final_state == [[0.25]]
    state=[[0.5]]
    result=replay(model,[0.0],(1,1);initial_state=state,trace=true)
    @test state == [[0.5]]
    result.final_state[1][1]=100.0
    @test result.state_trace == [[0.75]]
    model.layers[1].weights[1]=NaN
    @test_throws ArgumentError replay(model,[1.0],(1,1))
end
let
    m=ConvertedSNN([DenseLayer(1,1,[1.0],nothing,1.0)])
    @test replay(m,[1.0],(1,1);max_working_bytes=112).output == [1.0]
    @test_throws OutOfMemoryError replay(m,[1.0],(1,1);max_working_bytes=111)
    @test replay(m,[1.0],(1,1);trace=true,max_working_bytes=128).output == [1.0]
    @test_throws OutOfMemoryError replay(m,[1.0],(1,1);trace=true,max_working_bytes=127)
    @test isempty(replay(m,Float64[],(2^53,0);trace=true,max_working_bytes=16).output)
    @test_throws OutOfMemoryError replay(m,Float64[],(0,2^30))
    @test_throws ArgumentError replay(m,[NaN],(1,1))
    @test_throws ArgumentError replay(m,[0.5],(1,1))
    @test_throws ArgumentError replay(m,[1.5],(1,1);binary_inputs=false)
    @test_throws ArgumentError replay(m,[1.0],(-1,1))
    @test_throws ArgumentError replay(m,[1.0],(1,1);max_working_bytes=0)
    @test_throws ArgumentError replay(m,[1.0],(1,1);initial_state=Vector{Float64}[])
    @test_throws ArgumentError replay(m,[1.0],(1,1);initial_state=[[0.0,0.0]])
    @test_throws ArgumentError replay(m,[1.0],(1,1);initial_state=[[NaN]])
    @test_throws ArgumentError classify(m,ReplayResult([NaN],[],[],[]),1)
end
@test_throws ArgumentError DenseLayer(0,1,Float64[],nothing,1.0)
@test_throws ArgumentError DenseLayer(1,1,[1.0],[0.0,0.0],1.0)
@test_throws ArgumentError DenseLayer(1,1,[1.0],nothing,NaN)
@test_throws ArgumentError DenseLayer(1,1,[1.0],nothing,0.0)
@test_throws ArgumentError DenseLayer(1,1,[Inf],nothing,1.0)
@test_throws ArgumentError DenseLayer(1,1,[1.0],[Inf],1.0)
@test_throws ArgumentError DenseLayer(1,1,[1.0],nothing,1.0;initial_fraction=Inf)
@test_throws OutOfMemoryError DenseLayer(1,1,[1.0],nothing,1.0;max_working_bytes=15)
@test_throws ArgumentError ConvertedSNN(DenseLayer[])
@test_throws ArgumentError ConvertedSNN([DenseLayer(1,1,[1.0],nothing,1.0)];output_mode=:unknown)
@test_throws ArgumentError ConvertedSNN([DenseLayer(1,1,[1.0],nothing,1.0),DenseLayer(2,1,[1.0,1.0],nothing,1.0)])
let
    maximum=prevfloat(Inf)
    m=ConvertedSNN([DenseLayer(1,1,[maximum],[maximum],1.0)])
    state=[[0.0]]
    @test_throws OverflowError replay(m,[1.0],(1,1);initial_state=state)
    @test state == [[0.0]]
    shifted=ConvertedSNN([DenseLayer(1,1,[1.0],nothing,maximum;initial_fraction=maximum)])
    @test_throws OverflowError replay(shifted,[0.0],(1,1))
end
""")
    caller = tmp_path / "replay.jl"
    caller.write_text("\n".join(program))
    process = subprocess.run(
        [
            str(binaries[-1]),
            "--startup-file=no",
            "--history-file=no",
            "--check-bounds=yes",
            str(caller),
            str(api),
        ],
        capture_output=True,
        check=True,
        timeout=120,
    )
    assert cases == 48
    assert process.stdout == b"".join(expected)
