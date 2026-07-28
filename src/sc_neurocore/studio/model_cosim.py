# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Selected-model real RTL co-simulation

"""Run bit-exact C-reference versus Icarus RTL traces for one selected model."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # nosec B404
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.compiler.c_fixed_emitter import signed_q
from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.studio.model_compile_configuration import (
    ResolvedModelCompileConfiguration,
)

STUDIO_COSIM_PARITY_SCHEMA_VERSION = "studio.cosim-parity.v1"
BIT_TRUE_COSIM_INTEGRATORS = frozenset({"euler", "map"})
_TOOL_NAMES = ("gcc", "iverilog", "vvp")


@dataclass(frozen=True, slots=True)
class ModelCosimExecution:
    """Public parity report plus complete private artifacts for job custody."""

    reference_source: str
    reference_trace: list[list[int]]
    report: dict[str, object]
    rtl_source: str
    rtl_testbench: str
    rtl_trace: list[list[int]]


def run_model_cosim(
    configuration: ResolvedModelCompileConfiguration,
    *,
    current: float,
    n_steps: int,
) -> ModelCosimExecution:
    """Compile and compare real C-reference and RTL state traces cycle by cycle."""

    if configuration.integrator not in BIT_TRUE_COSIM_INTEGRATORS:
        supported = ", ".join(sorted(BIT_TRUE_COSIM_INTEGRATORS))
        raise ValueError(
            f"Bit-exact Studio co-simulation supports integrators {supported}; "
            f"got {configuration.integrator!r}."
        )
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError("Studio model co-simulation current must be a finite number.")
    current_float = float(current)
    if not (-float("inf") < current_float < float("inf")):
        raise ValueError("Studio model co-simulation current must be a finite number.")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int) or not 1 <= n_steps <= 2048:
        raise ValueError("Studio model co-simulation n_steps must be between 1 and 2048.")

    q_format = configuration.q_format
    if q_format.total_bits not in {8, 16, 32}:
        raise ValueError("Bit-exact Studio co-simulation requires an 8, 16 or 32-bit Q-format.")
    tools = {name: _resolve_tool(name) for name in _TOOL_NAMES}
    missing = [name for name, path in tools.items() if path is None]
    if missing:
        raise RuntimeError(f"Studio model co-simulation tools unavailable: {', '.join(missing)}.")

    equation_neuron = configuration.neuron.to_equation_neuron()
    rtl_source = configuration.to_verilog()
    reference_source = generate_bittrue_kernel_from_neuron(
        equation_neuron,
        configuration.module_name,
        data_width=q_format.total_bits,
        fraction=q_format.fraction_bits,
    )
    testbench = _rtl_testbench(
        equation_neuron,
        configuration.module_name,
        current_float,
        n_steps,
        q_format.total_bits,
        q_format.fraction_bits,
    )
    q = Q88(data_width=q_format.total_bits, fraction=q_format.fraction_bits)
    current_q = signed_q(q, current_float)
    c_source = (
        reference_source
        + "\n"
        + _c_main(
            equation_neuron,
            configuration.module_name,
            current_q,
            n_steps,
            q_format.total_bits,
        )
    )

    with tempfile.TemporaryDirectory(prefix="sc_studio_cosim_") as temp_dir:
        root = Path(temp_dir)
        rtl_path = root / "model.v"
        testbench_path = root / "tb.v"
        sim_path = root / "rtl_sim"
        c_path = root / "reference.c"
        reference_path = root / "reference"
        rtl_path.write_text(rtl_source, encoding="utf-8")
        testbench_path.write_text(testbench, encoding="utf-8")
        c_path.write_text(c_source, encoding="utf-8")

        _run_checked(
            [
                str(tools["iverilog"]),
                "-g2012",
                "-o",
                str(sim_path),
                str(rtl_path),
                str(testbench_path),
            ],
            timeout_seconds=60,
        )
        rtl_output = _run_checked([str(tools["vvp"]), str(sim_path)], timeout_seconds=60).stdout
        _run_checked(
            [str(tools["gcc"]), "-O2", "-std=c11", "-o", str(reference_path), str(c_path)],
            timeout_seconds=60,
        )
        reference_output = _run_checked([str(reference_path)], timeout_seconds=60).stdout

    signal_names = _signal_names(equation_neuron)
    rtl_trace = _parse_trace(rtl_output, n_steps=n_steps, n_signals=len(signal_names), label="RTL")
    reference_trace = _parse_trace(
        reference_output,
        n_steps=n_steps,
        n_signals=len(signal_names),
        label="reference",
    )
    bit_exact = rtl_trace == reference_trace
    first_mismatch = _first_mismatch(rtl_trace, reference_trace, signal_names)
    report: dict[str, object] = {
        "bit_exact": bit_exact,
        "configuration": configuration.to_public_dict(),
        "first_mismatch": first_mismatch,
        "module_name": configuration.module_name,
        "reference": {
            "kind": "generated_bit_true_c",
            "source_sha256": _sha256_text(c_source),
            "trace_sha256": _sha256_json(reference_trace),
        },
        "rtl": {
            "kind": "iverilog_vvp",
            "source_sha256": _sha256_text(rtl_source),
            "trace_sha256": _sha256_json(rtl_trace),
        },
        "sample_count": n_steps,
        "schema_version": STUDIO_COSIM_PARITY_SCHEMA_VERSION,
        "signals": signal_names,
        "status": "completed",
        "stimulus": {"current": current_float, "current_q": current_q, "n_steps": n_steps},
        "tools": {name: _tool_version(str(path), name) for name, path in tools.items()},
    }
    return ModelCosimExecution(
        reference_source=c_source,
        reference_trace=reference_trace,
        report=report,
        rtl_source=rtl_source,
        rtl_testbench=testbench,
        rtl_trace=rtl_trace,
    )


def _resolve_tool(name: str) -> str | None:
    if name not in _TOOL_NAMES:
        raise ValueError(f"Unsupported Studio co-simulation tool {name!r}.")
    return shutil.which(name)


def _run_checked(command: list[str], *, timeout_seconds: float) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(  # nosec B603
            command,
            capture_output=True,
            check=False,
            shell=False,
            text=True,
            timeout=timeout_seconds,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            f"Studio co-simulation command failed: {Path(command[0]).name}."
        ) from exc
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().replace("\n", " ")[:500]
        raise RuntimeError(
            f"Studio co-simulation command {Path(command[0]).name!r} exited "
            f"{completed.returncode}: {detail}"
        )
    return completed


def _tool_version(path: str, name: str) -> str:
    argument = "-V" if name in {"iverilog", "vvp"} else "--version"
    try:
        completed = _run_checked([path, argument], timeout_seconds=5)
    except RuntimeError:
        return "available-version-unreported"
    lines = (completed.stdout + "\n" + completed.stderr).strip().splitlines()
    return lines[0][:200] if lines else "available-version-unreported"


def _signal_names(neuron: EquationNeuron) -> list[str]:
    return [
        "spike_out",
        *(f"{sanitize_ident(name, context='state variable')}_out" for name in neuron.equations),
    ]


def _rtl_testbench(
    neuron: EquationNeuron,
    module_name: str,
    current: float,
    n_steps: int,
    data_width: int,
    fraction: int,
) -> str:
    q = Q88(data_width=data_width, fraction=fraction)
    variables = [sanitize_ident(name, context="state variable") for name in neuron.equations]
    ports = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        f"    .I_t({q.encode_signed_literal(current)}),",
        "    .spike_out(spike_out),",
    ]
    ports.extend(f"    .{name}_out({name}_out)," for name in variables)
    ports[-1] = ports[-1].rstrip(",")
    wires = [f"wire signed [{data_width - 1}:0] {name}_out;" for name in variables]
    fmt = " ".join(["%0d", *("%0d" for _ in variables)])
    args = ", ".join(["$unsigned(spike_out)", *(f"$signed({name}_out)" for name in variables)])
    return "\n".join(
        [
            "`timescale 1ns/1ps",
            f"module tb_{module_name};",
            "reg clk; reg rst_n; wire spike_out;",
            *wires,
            f"{module_name} uut (",
            *ports,
            ");",
            "initial clk=0; always #5 clk=~clk;",
            "integer k;",
            "initial begin",
            "  rst_n=0;",
            "  #23; rst_n=1;",
            f"  for(k=0;k<{n_steps};k=k+1) begin",
            "    @(posedge clk); #1;",
            f'    $display("{fmt}", {args});',
            "  end",
            "  $finish;",
            "end",
            "endmodule",
        ]
    )


def _c_main(
    neuron: EquationNeuron,
    module_name: str,
    current_q: int,
    n_steps: int,
    data_width: int,
) -> str:
    variables = [sanitize_ident(name, context="state variable") for name in neuron.equations]
    fmt = " ".join(["%d", *("%lld" for _ in variables)])
    args = ", ".join(["spike", *(f"(long long)st.{name}_out" for name in variables)])
    return (
        "#include <stdio.h>\n"
        f"int main(void) {{ {module_name}_state_t st; {module_name}_reset(&st); "
        f"int{data_width}_t I = {current_q};\n"
        f"  for (int k = 0; k < {n_steps}; k++) {{ int spike = {module_name}_step(&st, I); "
        f'printf("{fmt}\\n", {args}); }}\n'
        "  return 0; }\n"
    )


def _parse_trace(text: str, *, n_steps: int, n_signals: int, label: str) -> list[list[int]]:
    rows: list[list[int]] = []
    for line in text.strip().splitlines():
        tokens = line.split()
        if len(tokens) == n_signals and all(token.lstrip("-").isdigit() for token in tokens):
            rows.append([int(token) for token in tokens])
    if len(rows) != n_steps:
        raise RuntimeError(
            f"Studio {label} co-simulation emitted {len(rows)} of {n_steps} trace rows."
        )
    return rows


def _first_mismatch(
    rtl_trace: list[list[int]],
    reference_trace: list[list[int]],
    signal_names: list[str],
) -> dict[str, object] | None:
    for cycle, (rtl_row, reference_row) in enumerate(
        zip(rtl_trace, reference_trace, strict=True), start=1
    ):
        if rtl_row == reference_row:
            continue
        mismatched = [
            name
            for name, rtl, ref in zip(signal_names, rtl_row, reference_row, strict=True)
            if rtl != ref
        ]
        return {
            "cycle": cycle,
            "reference": dict(zip(signal_names, reference_row, strict=True)),
            "rtl": dict(zip(signal_names, rtl_row, strict=True)),
            "signals": mismatched,
        }
    return None


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()
