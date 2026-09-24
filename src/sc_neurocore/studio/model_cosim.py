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
import tempfile
from dataclasses import dataclass
from pathlib import Path

from sc_neurocore.compiler.c_fixed_emitter import signed_q
from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    c_word_type,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.hdl_gen._ident import sanitize_ident
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.studio.bit_true_execution import (
    NATIVE_TOOL_NAMES,
    native_tool_version,
    resolve_native_tool,
    run_native_command,
)
from sc_neurocore.studio.model_compile_configuration import (
    ResolvedModelCompileConfiguration,
)

STUDIO_COSIM_PARITY_SCHEMA_VERSION = "studio.cosim-parity.v1"
BIT_TRUE_COSIM_INTEGRATORS = frozenset({"euler", "map"})
COSIM_BOUNDARY = "rtl_vs_bittrue"
COSIM_BOUNDARY_STATEMENT = (
    "the Icarus Verilog RTL against the generated bit-true C kernel, both at "
    "configuration.numeric_contract; the floating-point model is not part of this "
    "comparison, and equal traces hold only for the stimuli listed"
)
COSIM_NOT_COVERED = (
    "a state landing exactly on the threshold, which needs a model-specific stimulus",
    "any stimulus other than the requested and stress schedules",
)
_TOOL_NAMES = NATIVE_TOOL_NAMES

# The native-tool helpers are shared with the bit-true precision comparison
# (:mod:`sc_neurocore.studio.bit_true_execution`); the module-level names stay
# so the process task and its tests address one resolution point.
_resolve_tool = resolve_native_tool
_run_checked = run_native_command
_tool_version = native_tool_version


@dataclass(frozen=True, slots=True)
class StimulusPhase:
    """One stretch of a co-simulation schedule.

    ``word`` is held on the input for ``steps`` cycles, after a reset of the
    neuron to its initial state when ``reset_before`` is set.
    """

    word: int
    steps: int
    reset_before: bool = False

    def to_public_dict(self) -> dict[str, object]:
        """Return the JSON projection."""
        return {"input_q": self.word, "steps": self.steps, "reset_before": self.reset_before}


def stress_schedule(current_q: int, data_width: int) -> tuple[StimulusPhase, ...]:
    """Return the fixed stress schedule run after the requested one.

    It holds the most negative and then the most positive input word, so the
    accumulate commit saturates in both directions; resets the neuron mid-run;
    replays the requested input from the initial state; and ends at zero input.
    Every phase starts after the requested run, from a reset neuron.
    """
    lowest = -(1 << (data_width - 1))
    highest = (1 << (data_width - 1)) - 1
    return (
        StimulusPhase(lowest, 16, reset_before=True),
        StimulusPhase(highest, 16),
        StimulusPhase(current_q, 16, reset_before=True),
        StimulusPhase(0, 8),
    )


@dataclass(frozen=True, slots=True)
class ModelCosimExecution:
    """Public parity report plus complete private artifacts for job custody."""

    reference_source: str
    reference_trace: list[list[int]]
    report: dict[str, object]
    rtl_source: str
    rtl_testbench: str
    rtl_trace: list[list[int]]
    stress_reference_trace: list[list[int]]
    stress_rtl_trace: list[list[int]]


def run_model_cosim(
    configuration: ResolvedModelCompileConfiguration,
    *,
    current: float,
    n_steps: int,
) -> ModelCosimExecution:
    """Compile and compare real C-reference and RTL state traces cycle by cycle.

    The requested constant ``current`` runs for ``n_steps`` cycles, then the
    :func:`stress_schedule` runs in the same simulation; ``bit_exact`` holds
    only when both traces agree. ``current`` must lie inside the Q-format's
    range, since an input word outside it would wrap.
    """
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

    q = Q88(data_width=q_format.total_bits, fraction=q_format.fraction_bits)
    if not q.min_value <= current_float <= q.max_value:
        raise ValueError(
            f"Studio model co-simulation current {current_float!r} is outside "
            f"{q_format.q_label} [{q.min_value!r}, {q.max_value!r}]; its input word would wrap."
        )
    current_q = signed_q(q, current_float)
    requested = (StimulusPhase(current_q, n_steps),)
    stress = stress_schedule(current_q, q_format.total_bits)
    stress_steps = sum(phase.steps for phase in stress)

    equation_neuron = configuration.neuron.to_equation_neuron()
    rtl_source = configuration.to_verilog()
    simulation = simulate_phases(
        equation_neuron,
        rtl_source,
        configuration.module_name,
        requested + stress,
        data_width=q_format.total_bits,
        fraction=q_format.fraction_bits,
        tools=tools,
    )
    rtl_output, reference_output = simulation.rtl_output, simulation.reference_output
    testbench, c_source = simulation.testbench, simulation.c_source

    signal_names = _signal_names(equation_neuron)
    total = n_steps + stress_steps
    rtl_rows = _parse_trace(rtl_output, n_steps=total, n_signals=len(signal_names), label="RTL")
    reference_rows = _parse_trace(
        reference_output,
        n_steps=total,
        n_signals=len(signal_names),
        label="reference",
    )
    rtl_trace, stress_rtl_trace = rtl_rows[:n_steps], rtl_rows[n_steps:]
    reference_trace, stress_reference_trace = reference_rows[:n_steps], reference_rows[n_steps:]
    first_mismatch = _first_mismatch(rtl_trace, reference_trace, signal_names)
    stress_mismatch = _first_mismatch(stress_rtl_trace, stress_reference_trace, signal_names)
    report: dict[str, object] = {
        "bit_exact": first_mismatch is None and stress_mismatch is None,
        "boundary": {
            "compared": COSIM_BOUNDARY,
            "statement": COSIM_BOUNDARY_STATEMENT,
            "not_covered": list(COSIM_NOT_COVERED),
        },
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
        "stress": {
            "bit_exact": stress_mismatch is None,
            "first_mismatch": stress_mismatch,
            "reference_trace_sha256": _sha256_json(stress_reference_trace),
            "rtl_trace_sha256": _sha256_json(stress_rtl_trace),
            "sample_count": stress_steps,
            "schedule": [phase.to_public_dict() for phase in stress],
        },
        "tools": {name: _tool_version(str(path), name) for name, path in tools.items()},
    }
    return ModelCosimExecution(
        reference_source=c_source,
        reference_trace=reference_trace,
        report=report,
        rtl_source=rtl_source,
        rtl_testbench=testbench,
        rtl_trace=rtl_trace,
        stress_reference_trace=stress_reference_trace,
        stress_rtl_trace=stress_rtl_trace,
    )


@dataclass(frozen=True, slots=True)
class PhaseSimulation:
    """What one RTL and generated-kernel run printed, and the sources it ran."""

    testbench: str
    c_source: str
    rtl_output: str
    reference_output: str


def simulate_phases(
    neuron: EquationNeuron,
    rtl_source: str,
    module_name: str,
    phases: tuple[StimulusPhase, ...],
    *,
    data_width: int,
    fraction: int,
    overflow: str = "saturate",
    rounding: str = "truncate",
    tools: dict[str, str | None] | None = None,
) -> PhaseSimulation:
    """Run ``rtl_source`` under Icarus and the generated C kernel through ``phases``.

    ``rtl_source`` must be the RTL of ``neuron`` compiled at the same word,
    overflow and rounding the kernel is generated with. Each prints one row per
    step: the spike and every state word.

    Raises
    ------
    ValueError
        No bit-true kernel mirrors ``neuron`` at this configuration.
    RuntimeError
        A tool is unavailable or a command fails.
    """
    resolved = tools or {name: _resolve_tool(name) for name in _TOOL_NAMES}
    missing = [name for name, path in resolved.items() if path is None]
    if missing:
        raise RuntimeError(f"Studio model co-simulation tools unavailable: {', '.join(missing)}.")
    reference_source = generate_bittrue_kernel_from_neuron(
        neuron,
        module_name,
        data_width=data_width,
        fraction=fraction,
        overflow=overflow,
        rounding=rounding,
    )
    testbench = _rtl_testbench(neuron, module_name, phases, data_width)
    c_source = reference_source + "\n" + _c_main(neuron, module_name, phases, data_width)
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
                str(resolved["iverilog"]),
                "-g2012",
                "-o",
                str(sim_path),
                str(rtl_path),
                str(testbench_path),
            ],
            timeout_seconds=60,
        )
        rtl_output = _run_checked([str(resolved["vvp"]), str(sim_path)], timeout_seconds=60).stdout
        _run_checked(
            [str(resolved["gcc"]), "-O2", "-std=c11", "-o", str(reference_path), str(c_path)],
            timeout_seconds=60,
        )
        reference_output = _run_checked([str(reference_path)], timeout_seconds=60).stdout
    return PhaseSimulation(
        testbench=testbench,
        c_source=c_source,
        rtl_output=rtl_output,
        reference_output=reference_output,
    )


def _signal_names(neuron: EquationNeuron) -> list[str]:
    return [
        "spike_out",
        *(f"{sanitize_ident(name, context='state variable')}_out" for name in neuron.equations),
    ]


def _rtl_testbench(
    neuron: EquationNeuron,
    module_name: str,
    phases: tuple[StimulusPhase, ...],
    data_width: int,
) -> str:
    """Drive the RTL through ``phases``, printing one row per rising clock edge.

    A reset pulse is asynchronous: it lands between two edges, after the row
    of the previous cycle is printed and before the next edge.
    """
    variables = [sanitize_ident(name, context="state variable") for name in neuron.equations]
    ports = [
        "    .clk(clk),",
        "    .rst_n(rst_n),",
        "    .I_t(I_drive),",
        "    .spike_out(spike_out),",
    ]
    ports.extend(f"    .{name}_out({name}_out)," for name in variables)
    ports[-1] = ports[-1].rstrip(",")
    wires = [f"wire signed [{data_width - 1}:0] {name}_out;" for name in variables]
    fmt = " ".join(["%0d", *("%0d" for _ in variables)])
    args = ", ".join(["$unsigned(spike_out)", *(f"$signed({name}_out)" for name in variables)])
    mask = (1 << data_width) - 1
    body: list[str] = []
    for phase in phases:
        if phase.reset_before:
            body.append("  rst_n=0; #1; rst_n=1;")
        body.extend(
            [
                f"  I_drive = {data_width}'sh{phase.word & mask:x};",
                f"  for(k=0;k<{phase.steps};k=k+1) begin",
                "    @(posedge clk); #1;",
                f'    $display("{fmt}", {args});',
                "  end",
            ]
        )
    return "\n".join(
        [
            "`timescale 1ns/1ps",
            f"module tb_{module_name};",
            "reg clk; reg rst_n; wire spike_out;",
            f"reg signed [{data_width - 1}:0] I_drive;",
            *wires,
            f"{module_name} uut (",
            *ports,
            ");",
            "initial clk=0; always #5 clk=~clk;",
            "integer k;",
            "initial begin",
            "  I_drive = 0;",
            "  rst_n=0;",
            "  #23; rst_n=1;",
            *body,
            "  $finish;",
            "end",
            "endmodule",
        ]
    )


def _c_main(
    neuron: EquationNeuron,
    module_name: str,
    phases: tuple[StimulusPhase, ...],
    data_width: int,
) -> str:
    """Step the generated C kernel through ``phases``, printing one row per step."""
    variables = [sanitize_ident(name, context="state variable") for name in neuron.equations]
    fmt = " ".join(["%d", *("%lld" for _ in variables)])
    args = ", ".join(["spike", *(f"(long long)st.{name}_out" for name in variables)])
    word_type = c_word_type(data_width)
    body: list[str] = []
    for phase in phases:
        if phase.reset_before:
            body.append(f"  {module_name}_reset(&st);\n")
        body.append(
            f"  I = ({word_type})({phase.word}LL);\n"
            f"  for (int k = 0; k < {phase.steps}; k++) {{ int spike = {module_name}_step(&st, I); "
            f'printf("{fmt}\\n", {args}); }}\n'
        )
    return (
        "#include <stdio.h>\n"
        f"int main(void) {{ {module_name}_state_t st; {module_name}_reset(&st); "
        f"{word_type} I;\n" + "".join(body) + "  return 0; }\n"
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
