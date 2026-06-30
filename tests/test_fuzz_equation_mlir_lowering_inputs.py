# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Property-based fuzz tests for equation and MLIR lowering

"""Property-based fuzz tests for equation-to-HDL and MLIR lowering inputs."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sc_neurocore.compiler.equation_compiler import equation_to_fpga
from sc_neurocore.compiler.mlir_emitter import MLIREmitter, MLIRNode
from sc_neurocore.hdl_gen._ident import sanitize_ident

_VALID_IDENT = st.from_regex(r"[A-Za-z_][A-Za-z0-9_]{0,30}", fullmatch=True).filter(
    lambda value: _is_valid_identifier(value)
)
_MALICIOUS_IDENT = st.text(min_size=1, max_size=80).filter(
    lambda value: not _is_valid_identifier(value)
)
_MALICIOUS_SIGNAL = st.text(min_size=1, max_size=80).filter(
    lambda value: not _is_valid_signal_name(value)
)


def _is_valid_identifier(value: str) -> bool:
    try:
        sanitize_ident(value)
    except ValueError:
        return False
    return True


def _is_valid_signal_name(value: str) -> bool:
    return _is_valid_identifier(value[1:] if value.startswith("%") else value)


@given(module_name=_VALID_IDENT)
@settings(max_examples=80, deadline=None)
def test_fuzz_equation_to_fpga_accepts_valid_module_names(module_name: str) -> None:
    _, verilog = equation_to_fpga(
        "dv/dt = I",
        init={"v": 0.0},
        dt=1.0,
        module_name=module_name,
    )

    # `dv/dt = I` declares no parameters, so the header has no `#(...)` clause (an
    # empty one is malformed Verilog and is dropped); a parameterised neuron would emit
    # `module <name> #(`. Accept either valid header form.
    assert f"module {module_name} (" in verilog or f"module {module_name} #(" in verilog
    assert "endmodule" in verilog


@given(module_name=_MALICIOUS_IDENT)
@settings(max_examples=120, deadline=None)
def test_fuzz_equation_to_fpga_rejects_invalid_module_names(module_name: str) -> None:
    with pytest.raises(ValueError, match="module name"):
        equation_to_fpga(
            "dv/dt = I",
            init={"v": 0.0},
            dt=1.0,
            module_name=module_name,
        )


@given(
    param_name=_VALID_IDENT,
    value=st.floats(
        min_value=-128.0,
        max_value=127.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(max_examples=80, deadline=None)
def test_fuzz_equation_to_fpga_accepts_valid_parameter_names(param_name: str, value: float) -> None:
    _, verilog = equation_to_fpga(
        f"dv/dt = {param_name}",
        params={param_name: value},
        init={"v": 0.0},
        dt=1.0,
        module_name="param_lowering",
    )

    assert f"P_{param_name.upper()}" in verilog


@given(param_name=_MALICIOUS_IDENT)
@settings(max_examples=120, deadline=None)
def test_fuzz_equation_to_fpga_rejects_invalid_parameter_names(param_name: str) -> None:
    with pytest.raises(ValueError, match="parameter name"):
        equation_to_fpga(
            "dv/dt = I",
            params={param_name: 1.0},
            init={"v": 0.0},
            dt=1.0,
        )


@given(module_name=_VALID_IDENT, signal_name=_VALID_IDENT)
@settings(max_examples=80, deadline=None)
def test_fuzz_mlir_emitter_sanitises_final_output(module_name: str, signal_name: str) -> None:
    emitter = MLIREmitter(module_name)
    emitter.nodes.append(MLIRNode("comb.and", ["%lhs", "%rhs"], signal_name, {}))

    mlir = emitter.generate()

    assert f"hw.module @{module_name}" in mlir
    assert f"hw.output %{signal_name} : i1" in mlir


@given(module_name=_MALICIOUS_IDENT)
@settings(max_examples=120, deadline=None)
def test_fuzz_mlir_emitter_rejects_invalid_module_names(module_name: str) -> None:
    emitter = MLIREmitter(module_name)
    emitter.emit_and("%lhs", "%rhs")

    with pytest.raises(ValueError, match="module name"):
        emitter.generate()


@given(signal_name=_MALICIOUS_SIGNAL)
@settings(max_examples=120, deadline=None)
def test_fuzz_mlir_emitter_rejects_invalid_signal_inputs(signal_name: str) -> None:
    emitter = MLIREmitter("safe_top")
    emitter.nodes.append(MLIRNode("comb.xor", [signal_name, "%rhs"], "%out", {}))

    with pytest.raises(ValueError, match="signal name"):
        emitter.generate()


@given(signal_name=_MALICIOUS_SIGNAL)
@settings(max_examples=120, deadline=None)
def test_fuzz_mlir_emitter_rejects_invalid_signal_outputs(signal_name: str) -> None:
    emitter = MLIREmitter("safe_top")
    emitter.nodes.append(MLIRNode("comb.xor", ["%lhs", "%rhs"], signal_name, {}))

    with pytest.raises(ValueError, match="signal name"):
        emitter.generate()


@given(instance_name=_MALICIOUS_IDENT, module_name=_MALICIOUS_IDENT)
@settings(max_examples=120, deadline=None)
def test_fuzz_mlir_emitter_rejects_invalid_instance_attributes(
    instance_name: str, module_name: str
) -> None:
    emitter = MLIREmitter("safe_top")
    emitter.nodes.append(
        MLIRNode(
            "hw.instance",
            [],
            "%out",
            {"sym_name": instance_name, "module": module_name},
        )
    )

    with pytest.raises(ValueError, match="(signal name|module name)"):
        emitter.generate()
