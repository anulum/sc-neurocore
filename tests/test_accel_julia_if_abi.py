# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Julia dense IF C callback and rooted-owner acceptance

"""Invoke real Julia C callbacks against public NumPy golden bits and raw admission failures."""

import subprocess
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from sc_neurocore.conversion import ConvertedSNN
from sc_neurocore.conversion.if_parameters import OutputMode


def values(array: npt.NDArray[np.float64]) -> str:
    """Encode real coefficient/frame storage without changing IEEE-754 bits."""
    return (
        "reinterpret(Float64,UInt64["
        + ",".join(str(int(v)) for v in array.ravel().view(np.uint64))
        + "])"
    )


@pytest.mark.parametrize("runtime", ["1.11", "1.13"])
def test_julia_c_callbacks_complete_buffers_and_raw_refusal(tmp_path: Path, runtime: str) -> None:
    """Compile actual C callbacks in each native runtime, retain owners across GC, and compare all bits."""
    binaries = sorted((Path.home() / ".julia/juliaup").glob(f"julia-{runtime}.*/bin/julia"))
    assert binaries, f"Julia {runtime} native runtime required"
    root = Path(__file__).resolve().parents[1]
    api = root / "src/sc_neurocore/accel/julia/conversion/ann_to_snn_native.jl"
    program = [
        "include(ARGS[1])",
        "const N=AnnToSnnNative",
        "const L=N.LayerSpec",
        "const R=N.ReplayRequest",
        "const V=N.BufferView",
        "const replay_c=@cfunction(N.sc_if_replay,Cint,(Ptr{R},Ptr{Ptr{Cvoid}}))",
        "const buffer_c=@cfunction(N.sc_if_buffer,Cint,(Ptr{UInt},UInt32,UInt,Ptr{V}))",
        "const free_c=@cfunction(N.sc_if_free,Cvoid,(Ptr{UInt},))",
        "@assert N.sc_if_abi_version()==1",
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
                    frames = rng.random((steps, batch, 3))
                    if binary:
                        frames = (frames < 0.5).astype(np.float64)
                    initial = (
                        [rng.normal(0, 0.5, (batch, 4)), rng.normal(0, 0.5, (batch, 2))]
                        if seed % 2
                        else None
                    )
                    model = ConvertedSNN(
                        weights,
                        biases,
                        [0.75, 1.25],
                        T=7,
                        output_mode=output_mode,
                        layer_membrane_fractions=[0.0, 0.5],
                    )
                    result = model.replay(
                        frames,
                        initial_state=initial,
                        trace=True,
                        binary_inputs=binary,
                        backend="numpy",
                    )
                    groups = [
                        (result.output,),
                        result.final_state,
                        result.state_trace,
                        result.spike_trace,
                    ]
                    expected.extend(
                        array.astype("<f8", copy=False).tobytes()
                        for group in groups
                        for array in group
                    )
                    program += [
                        "let",
                        "weights=[" + ",".join(values(w) for w in weights) + "]",
                        "biases=["
                        + ",".join("Float64[]" if b is None else values(b) for b in biases)
                        + "]",
                        "states="
                        + (
                            "[Float64[],Float64[]]"
                            if initial is None
                            else "[" + ",".join(values(s) for s in initial) + "]"
                        ),
                        f"frames={values(frames)}",
                        "specs=L[L(UInt(4),UInt(3),pointer(weights[1]),pointer(biases[1]),UInt(length(biases[1])),0.75,0.0,pointer(states[1]),UInt(length(states[1]))),"
                        "L(UInt(2),UInt(4),pointer(weights[2]),pointer(biases[2]),UInt(length(biases[2])),1.25,0.5,pointer(states[2]),UInt(length(states[2])))]",
                    ]
                    flags = (
                        1
                        | (2 if binary else 0)
                        | (4 if mode == "linear" else 0)
                        | (8 if initial is not None else 0)
                    )
                    program += [
                        f"request=Ref(R(UInt32(1),UInt32({flags}),pointer(specs),UInt(2),pointer(frames),UInt(length(frames)),UInt({steps}),UInt({batch}),UInt(1<<20)))",
                        "owner=Ref(Ptr{Cvoid}(C_NULL))",
                        "GC.@preserve weights biases frames states specs request owner begin",
                        "@assert ccall(replay_c,Cint,(Ref{R},Ref{Ptr{Cvoid}}),request,owner)==0",
                        "end",
                        "fill!(frames,0);fill!(weights[1],99);fill!(states[1],99);GC.gc()",
                        "try",
                    ]
                    for kind, group in enumerate(groups):
                        for index, array in enumerate(group):
                            program += [
                                "view=Ref(V(Ptr{Float64}(C_NULL),UInt(0)))",
                                f"@assert ccall(buffer_c,Cint,(Ptr{{UInt}},UInt32,UInt,Ref{{V}}),owner[],UInt32({kind}),UInt({index}),view)==0",
                                f"@assert view[].len=={array.size}",
                                "for i in 1:Int(view[].len);write(stdout,htol(reinterpret(UInt64,unsafe_load(view[].data,i))));end",
                            ]
                    program += [
                        "view=Ref(V(Ptr{Float64}(16),UInt(999)))",
                        "@assert ccall(buffer_c,Cint,(Ptr{UInt},UInt32,UInt,Ref{V}),owner[],UInt32(99),UInt(0),view)==-1",
                        "@assert view[].len==999 && UInt(view[].data)==16",
                        "finally",
                        "ccall(free_c,Cvoid,(Ptr{UInt},),owner[])",
                        "end",
                        "end",
                    ]
    program.append(r"""
let
    weight=[1.0];frame=[1.0]
    base=L(UInt(1),UInt(1),pointer(weight),Ptr{Float64}(C_NULL),UInt(0),1.0,0.0,Ptr{Float64}(C_NULL),UInt(0))
    specs=[base]
    good=R(UInt32(1),UInt32(3),pointer(specs),UInt(1),pointer(frame),UInt(1),UInt(1),UInt(1),UInt(128))
    GC.@preserve weight frame specs begin
        for (request,status) in (
            (R(UInt32(2),good.flags,good.layers,good.layer_count,good.frames,good.frames_len,good.steps,good.batch,good.max_working_bytes),-1),
            (R(good.version,UInt32(32),good.layers,good.layer_count,good.frames,good.frames_len,good.steps,good.batch,good.max_working_bytes),-1),
            (R(good.version,good.flags,good.layers,good.layer_count,good.frames,good.frames_len,good.steps,good.batch,UInt(127)),-2),
            (R(good.version,good.flags,good.layers,good.layer_count,good.frames,good.frames_len,good.steps,good.batch,UInt(0)),-1),
            (R(good.version,good.flags,good.layers,UInt(1)<<53,good.frames,good.frames_len,good.steps,good.batch,good.max_working_bytes),-2),
            (R(good.version,good.flags,good.layers,good.layer_count,Ptr{Float64}(1),good.frames_len,good.steps,good.batch,good.max_working_bytes),-1),
            (R(good.version,good.flags,Ptr{L}(C_NULL),good.layer_count,good.frames,good.frames_len,good.steps,good.batch,good.max_working_bytes),-1))
            owner=Ref(Ptr{Cvoid}(UInt(0x1234)))
            @assert ccall(replay_c,Cint,(Ref{R},Ref{Ptr{Cvoid}}),Ref(request),owner)==status
            @assert UInt(owner[])==0x1234
        end
        for invalid in (NaN,1.5,0.5)
            frame[1]=invalid;owner=Ref(Ptr{Cvoid}(UInt(0x1234)))
            @assert ccall(replay_c,Cint,(Ref{R},Ref{Ptr{Cvoid}}),Ref(good),owner)==-1
            @assert UInt(owner[])==0x1234
        end
        frame[1]=1;weight[1]=floatmax(Float64);bias=[floatmax(Float64)]
        specs[1]=L(UInt(1),UInt(1),pointer(weight),pointer(bias),UInt(1),1.0,0.0,Ptr{Float64}(C_NULL),UInt(0))
        overflow=R(good.version,good.flags,good.layers,good.layer_count,good.frames,good.frames_len,good.steps,good.batch,UInt(1024))
        owner=Ref(Ptr{Cvoid}(UInt(0x1234)))
        GC.@preserve bias begin
            @assert ccall(replay_c,Cint,(Ref{R},Ref{Ptr{Cvoid}}),Ref(overflow),owner)==-3
        end
        @assert UInt(owner[])==0x1234
        @assert ccall(replay_c,Cint,(Ptr{R},Ref{Ptr{Cvoid}}),C_NULL,owner)==-1
        @assert ccall(replay_c,Cint,(Ref{R},Ptr{Ptr{Cvoid}}),Ref(good),C_NULL)==-1
        ccall(free_c,Cvoid,(Ptr{UInt},),C_NULL)
    end
end
""")
    caller = tmp_path / "complete-native-abi.jl"
    caller.write_text("\n".join(program))
    response = subprocess.run(
        [
            str(binaries[-1]),
            "--startup-file=no",
            "--check-bounds=yes",
            "--depwarn=error",
            str(caller),
            str(api),
        ],
        check=False,
        capture_output=True,
        timeout=120,
    )
    (tmp_path / "native-stdout.bin").write_bytes(response.stdout)
    (tmp_path / "native-stderr.txt").write_bytes(response.stderr)
    assert response.returncode == 0, response.stderr.decode(errors="replace")
    expected_bytes = b"".join(expected)
    assert response.stdout == expected_bytes
    (tmp_path / "verified-response.bin").write_bytes(response.stdout)
    assert len(expected_bytes) == 34688
