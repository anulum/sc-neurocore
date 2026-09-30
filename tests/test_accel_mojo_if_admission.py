# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo public dense IF ownership and admission

"""Exercise actual native refusals and independent result/parameter ownership."""

import subprocess
from pathlib import Path
from sc_neurocore.accel.mojo.isa_baseline import pin_isa


def test_mojo_native_admission_ownership_and_overflow(tmp_path: Path) -> None:
    """Check exported constructors/replay/classify with real boundary values and states."""
    root = Path(__file__).resolve().parents[1]
    kernels = root / "src/sc_neurocore/accel/mojo/kernels"
    source = [
        r'''"""Exercise native dense IF refusal categories and buffer custody."""
from std.memory import bitcast
from ann_to_snn import replay, classify, ReplayResult
from ann_to_snn_parameters import DenseLayer, ConvertedSNN


def main() raises:
    """Run actual native admission, overflow and independent ownership assertions.

    Raises:
        Error: Unexpected acceptance, error category or ownership violation.
    """
    var nan = bitcast[DType.float64](UInt64(0x7FF8000000000000))
    var maximum = bitcast[DType.float64](UInt64(0x7FEFFFFFFFFFFFFF))
    var empty = List[Float64]()
    var empty_layers = List[DenseLayer]()
    var weights: List[Float64] = [1.0]
    var bias: List[Float64] = [0.25]
    var layers = List[DenseLayer]()
    layers.append(DenseLayer(1,1,weights,bias,1.0,0.5))
    var owned = ConvertedSNN(layers)
    weights[0] = 100
    bias[0] = 100
    layers[0].weights[0] = 100
    var frames: List[Float64] = [0.5]
    var states: List[List[Float64]] = [[0.5]]
    var r = replay(owned,frames,1,1,initial_state=states,use_initial_state=True,trace=True,binary_inputs=False)
    if r.output[0] != 1 or r.final_state[0][0] != 0.25 or states[0][0] != 0.5:
        raise Error("caller parameter/state ownership violation")
    r.final_state[0][0] = 100
    if r.state_trace[0][0] != 0.25:
        raise Error("result state aliases its trace")
    frames[0] = 0
    var unit: List[Float64] = [1.0]
    var current: List[Float64] = [0.5]
    var out_of_range: List[Float64] = [1.5]
    var invalid: List[Float64] = [nan]
    var invalid_state: List[List[Float64]] = [[nan]]
    var wrong_state: List[List[Float64]] = [[0,0]]
    var missing_state = List[List[Float64]]()
    var plain_layers = List[DenseLayer]()
    plain_layers.append(DenseLayer(1,1,unit,empty,1.0))
    var plain = ConvertedSNN(plain_layers)
    var no_trace = replay(plain,unit,1,1,max_working_bytes=112)
    var traced = replay(plain,unit,1,1,trace=True,max_working_bytes=128)
    if no_trace.output[0] != 1 or traced.output[0] != 1:
        raise Error("exact reservation response mismatch")
    var vacant = replay(plain,empty,9007199254740992,0,trace=True,max_working_bytes=16)
    if len(vacant.output) != 0:
        raise Error("empty batch response mismatch")
    var bad_bias: List[Float64] = [0,0]
    var bad_weights: List[Float64] = [nan]
    var large: List[Float64] = [maximum]
    var shifted_layers = List[DenseLayer]()
    shifted_layers.append(DenseLayer(1,1,unit,empty,maximum,maximum))
    var shifted = ConvertedSNN(shifted_layers)
    var overflow_layers = List[DenseLayer]()
    overflow_layers.append(DenseLayer(1,1,large,large,1.0))
    var overflow = ConvertedSNN(overflow_layers)
    var corrupted = plain.copy()
    corrupted.layers[0].weights[0] = nan
    var bad_width = plain.copy()
    bad_width.layers[0].outputs = 0
    var empty_result = ReplayResult()
    var result = ReplayResult()
    result.output.append(nan)
'''
    ]
    cases = [
        ("DenseLayer(0,1,empty,empty,1.0)", "IF invalid input"),
        ("DenseLayer(1,1,unit,bad_bias,1.0)", "IF invalid input"),
        ("DenseLayer(1,1,unit,empty,0.0)", "IF invalid input"),
        ("DenseLayer(1,1,bad_weights,empty,1.0)", "IF invalid input"),
        ("DenseLayer(1,1,unit,invalid,1.0)", "IF invalid input"),
        ("DenseLayer(1,1,unit,empty,1.0,nan)", "IF invalid input"),
        ("DenseLayer(1,1,unit,empty,1.0,max_working_bytes=15)", "IF resource limit"),
        ("DenseLayer(1,1,unit,empty,1.0,max_working_bytes=0)", "IF invalid input"),
        ("ConvertedSNN(empty_layers)", "IF invalid input"),
        ("ConvertedSNN(plain_layers,max_working_bytes=15)", "IF resource limit"),
        ("replay(plain,invalid,1,1)", "IF invalid input"),
        ("replay(plain,out_of_range,1,1,binary_inputs=False)", "IF invalid input"),
        ("replay(plain,current,1,1)", "IF invalid input"),
        ("replay(plain,unit,-1,1)", "IF invalid input"),
        ("replay(plain,unit,2,1)", "IF invalid input"),
        ("replay(plain,unit,1,1,max_working_bytes=0)", "IF invalid input"),
        ("replay(plain,unit,1,1,max_working_bytes=111)", "IF resource limit"),
        ("replay(plain,unit,1,1,trace=True,max_working_bytes=127)", "IF resource limit"),
        ("replay(plain,empty,0,1073741824)", "IF resource limit"),
        ("replay(plain,empty,0,9223372036854775807)", "IF resource limit"),
        ("replay(plain,empty,9223372036854775807,2)", "IF invalid input"),
        (
            "replay(plain,unit,1,1,initial_state=missing_state,use_initial_state=True)",
            "IF invalid input",
        ),
        (
            "replay(plain,unit,1,1,initial_state=wrong_state,use_initial_state=True)",
            "IF invalid input",
        ),
        (
            "replay(plain,unit,1,1,initial_state=invalid_state,use_initial_state=True)",
            "IF invalid input",
        ),
        ("replay(shifted,unit,1,1)", "IF overflow"),
        ("replay(overflow,unit,1,1)", "IF overflow"),
        ("replay(corrupted,unit,1,1)", "IF invalid input"),
        ("classify(plain,result,1)", "IF invalid input"),
        ("classify(plain,result,-1)", "IF invalid input"),
        ("classify(plain,result,2)", "IF invalid input"),
        ("classify(bad_width,empty_result,1)", "IF invalid input"),
    ]
    for i, (expression, expected) in enumerate(cases):
        source.append(
            f"    var refused{i} = False\n    try:\n        _ = {expression}\n"
            f'    except e:\n        if String(e) != "{expected}":\n'
            "            raise e^\n"
            f"        refused{i} = True\n    if not refused{i}:\n"
            f'        raise Error("native refusal {i} was accepted")'
        )
    source.append("""    var original = replay(plain,unit,1,1)
    if original.output[0] != 1:
        raise Error("copied model aliases original parameters")
    print("native admission and ownership passed")""")
    caller = tmp_path / "admission.mojo"
    caller.write_text("\n".join(source))
    executable = tmp_path / "admission"
    subprocess.run(
        pin_isa(
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
            ]
        ),
        capture_output=True,
        check=True,
        timeout=120,
    )
    result = subprocess.run([str(executable)], capture_output=True, check=True, timeout=30)
    assert result.stdout == b"native admission and ownership passed\n"
