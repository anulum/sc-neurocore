# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Studio network RTL run beside its bit-true model and the Studio run

"""Compile a lowered Studio network and check its RTL against two references.

Three spike rasters of the same network, step by step:

* the **RTL**, compiled by the hardware network compiler and run by Icarus
  Verilog with the Studio's drive on its input lanes;
* a **bit-true model** of that RTL in C: each neuron is the whole-neuron kernel
  the compiler's bit-true generator emits from the same equations, and the
  network loop reproduces the compiled interconnect — quantised weights,
  saturating accumulation, registered spikes and delay chains. The RTL and
  this model must agree on every step; a difference is a compiler defect;
* the **Studio's own run** in double precision, which the fixed-point hardware
  can only approach. Where they differ, the first differing step is reported,
  not hidden.

Every artefact is bound to the lowering's input digest: the RTL sources and the
model source are hashed and reported beside it.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sc_neurocore.compiler.intelligence.bit_true_kernel import (
    c_word_type,
    generate_bittrue_kernel_from_neuron,
)
from sc_neurocore.compiler.verilog_compiler_config import Q88
from sc_neurocore.nir_bridge.fpga_compilation_result import NetworkCompilationResult
from sc_neurocore.nir_bridge.fpga_compiler import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_neuron_rtl import _population_neuron
from sc_neurocore.nir_bridge.quantise_params import quantise_graph
from sc_neurocore.studio.network_graph import simulate_graph
from sc_neurocore.studio.network_hardware import LoweredNetwork

NETWORK_COSIM_SCHEMA_VERSION = "sc-neurocore.studio.network-cosim.v1"
HARDWARE_MODULE_NAME = "studio_network"
_TOOLS = ("iverilog", "vvp", "gcc")


class HardwareCosimUnavailable(RuntimeError):
    """The simulation tools the co-simulation needs are not installed."""


@dataclass(frozen=True, slots=True)
class NetworkCosim:
    """The three rasters of one network, and the digests that bind them."""

    input_sha256: str
    rtl_sha256: str
    model_sha256: str
    steps: int
    rtl_raster: tuple[str, ...]
    model_raster: tuple[str, ...]
    studio_raster: tuple[str, ...]

    @property
    def rtl_matches_model(self) -> bool:
        """Whether the RTL and its bit-true model spike identically on every step."""
        return self.rtl_raster == self.model_raster

    @property
    def studio_first_divergence(self) -> int | None:
        """The first step where the RTL and the Studio's run differ, if any."""
        for step, (rtl, studio) in enumerate(zip(self.rtl_raster, self.studio_raster)):
            if rtl != studio:
                return step
        return None

    def to_public_dict(self) -> dict[str, Any]:
        """Return the JSON receipt of this co-simulation."""
        return {
            "schema_version": NETWORK_COSIM_SCHEMA_VERSION,
            "input_sha256": self.input_sha256,
            "rtl_sha256": self.rtl_sha256,
            "model_sha256": self.model_sha256,
            "steps": self.steps,
            "rtl_matches_bit_true_model": self.rtl_matches_model,
            "studio_agreement": {
                "identical": self.studio_first_divergence is None,
                "first_divergent_step": self.studio_first_divergence,
            },
            "spike_counts": {
                "rtl": sum(row.count("1") for row in self.rtl_raster),
                "bit_true_model": sum(row.count("1") for row in self.model_raster),
                "studio": sum(row.count("1") for row in self.studio_raster),
            },
        }


def compile_lowered(lowered: LoweredNetwork) -> NetworkCompilationResult:
    """Compile a lowered network with the direct interconnect the model reproduces."""
    return compile_network_to_fpga(
        lowered.graph,
        module_name=HARDWARE_MODULE_NAME,
        data_width=lowered.data_width,
        fraction=lowered.fraction,
        interconnect="direct",
    )


def cosimulate(
    lowered: LoweredNetwork,
    graph: object,
    workdir: Path,
    *,
    steps: int | None = None,
    timeout_seconds: float = 300.0,
) -> NetworkCosim:
    """Run the compiled RTL, its bit-true model and the Studio for the same steps.

    Parameters
    ----------
    lowered:
        The lowering of ``graph``.
    graph:
        The Studio graph the lowering came from; the Studio's run uses it.
    workdir:
        An empty directory the sources, executables and outputs are written to.
    steps:
        Steps to run; the graph's own step count when omitted.

    Raises
    ------
    HardwareCosimUnavailable
        When Icarus Verilog or a C compiler is not installed.
    """
    missing = [tool for tool in _TOOLS if shutil.which(tool) is None]
    if missing:
        raise HardwareCosimUnavailable(
            f"network co-simulation needs {', '.join(missing)}, which is not installed"
        )
    run_steps = lowered.spec.n_steps if steps is None else steps
    compiled = compile_lowered(lowered)
    rtl_sources = _rtl_sources(compiled)
    model_source = _model_source(lowered, compiled, run_steps)
    rtl_raster = _run_rtl(lowered, compiled, rtl_sources, run_steps, workdir, timeout_seconds)
    model_raster = _run_model(model_source, workdir, timeout_seconds)
    return NetworkCosim(
        input_sha256=lowered.input_sha256(),
        rtl_sha256=_digest(rtl_sources),
        model_sha256=hashlib.sha256(model_source.encode("utf-8")).hexdigest(),
        steps=run_steps,
        rtl_raster=rtl_raster,
        model_raster=model_raster,
        studio_raster=_studio_raster(graph, lowered, run_steps),
    )


def _rtl_sources(compiled: NetworkCompilationResult) -> dict[str, str]:
    sources = {f"{HARDWARE_MODULE_NAME}.v": compiled.top_module}
    sources.update({f"sc_nir_{name}.v": text for name, text in compiled.neuron_modules.items()})
    sources.update({f"{name}.v": text for name, text in compiled.scnir_source_modules.items()})
    sources["weight_rom.v"] = compiled.weight_rom
    return sources


def _digest(sources: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(sources):
        digest.update(name.encode("utf-8") + b"\0" + sources[name].encode("utf-8") + b"\0")
    return digest.hexdigest()


def _lane_words(lowered: LoweredNetwork, compiled: NetworkCompilationResult) -> list[int]:
    """Return the fixed-point word on every external lane, in bus order."""
    q = Q88(data_width=lowered.data_width, fraction=lowered.fraction)
    currents = dict(lowered.drive_lanes)
    words: list[int] = []
    for entry in sorted(compiled.scnir_external_inputs, key=lambda item: item.offset):
        words.extend(
            int(round(value * (1 << q.fraction))) for value in currents[entry.source][: entry.width]
        )
    return words


def _run_rtl(
    lowered: LoweredNetwork,
    compiled: NetworkCompilationResult,
    sources: dict[str, str],
    steps: int,
    workdir: Path,
    timeout_seconds: float,
) -> tuple[str, ...]:
    rtl = workdir / "rtl"
    rtl.mkdir(parents=True)
    for name, text in sources.items():
        (rtl / name).write_text(text, encoding="utf-8")
    words = _lane_words(lowered, compiled)
    width = lowered.data_width
    lanes = max(1, len(words))
    flat = (
        ", ".join(
            f"{width}'sd{word}" if word >= 0 else f"-{width}'sd{-word}" for word in reversed(words)
        )
        or f"{width}'sd0"
    )
    neurons = lowered.graph.total_neurons
    (rtl / "tb.v").write_text(
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  reg clk = 0, rst_n = 0, en = 1;\n"
        f"  wire signed [{lanes * width - 1}:0] lanes = {{{flat}}};\n"
        f"  wire [{neurons - 1}:0] spikes;\n"
        "  integer step;\n"
        f"  {HARDWARE_MODULE_NAME} dut(.clk(clk), .rst_n(rst_n), .en(en), "
        ".I_ext_flat(lanes), .spike_bus(spikes));\n"
        "  always #5 clk = ~clk;\n"
        "  initial begin\n"
        "    #12 rst_n = 1;\n"
        f"    for (step = 0; step < {steps}; step = step + 1) begin\n"
        '      @(posedge clk); #1 $display("%b", spikes);\n'
        "    end\n"
        "    $finish;\n"
        "  end\n"
        "endmodule\n",
        encoding="utf-8",
    )
    binary = rtl / "network.vvp"
    subprocess.run(
        ["iverilog", "-g2012", "-o", str(binary), *sorted(str(path) for path in rtl.glob("*.v"))],
        check=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    output = subprocess.run(
        ["vvp", "-n", str(binary)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    ).stdout
    return tuple(line for line in output.splitlines() if re.fullmatch(r"[01]+", line))


def _model_source(lowered: LoweredNetwork, compiled: NetworkCompilationResult, steps: int) -> str:
    """Return a C program that runs the bit-true model of the compiled network."""
    width, fraction = lowered.data_width, lowered.fraction
    word = c_word_type(width)
    match = re.search(r"localparam integer ACC_WIDTH = (\d+);", compiled.top_module)
    acc_width = int(match.group(1)) if match else 64
    q = Q88(data_width=width, fraction=fraction)
    qgraph = quantise_graph(lowered.graph, q)
    offsets: dict[str, int] = {}
    cursor = 0
    for population in lowered.graph.populations:
        offsets[population.name] = cursor
        cursor += population.n_neurons
    lane_offsets = {entry.source: entry.offset for entry in compiled.scnir_external_inputs}
    words = _lane_words(lowered, compiled)

    parts = [
        f"#include <stdint.h>\n#include <stdio.h>\n#define NEURONS {cursor}\n#define STEPS {steps}\n"
    ]
    kernels: dict[str, str] = {}
    for population in lowered.graph.populations:
        if population.neuron_type not in kernels:
            kernels[population.neuron_type] = f"k_{population.neuron_type}"
            parts.append(
                generate_bittrue_kernel_from_neuron(
                    _population_neuron(population.neuron_type, population),
                    kernels[population.neuron_type],
                    data_width=width,
                    fraction=fraction,
                )
            )
    parts.append(
        "static int64_t wrap_acc(int64_t x) {\n"
        f"    const int64_t span = (int64_t)1 << {acc_width};\n"
        f"    int64_t y = x & (span - 1);\n"
        f"    return y >= (span >> 1) ? y - span : y;\n"
        "}\n"
        f"static {word} sat_word(int64_t x) {{\n"
        f"    const int64_t hi = ((int64_t)1 << {width - 1}) - 1, lo = -((int64_t)1 << {width - 1});\n"
        f"    return ({word})(x > hi ? hi : x < lo ? lo : x);\n"
        "}\n"
    )
    lanes = ", ".join(str(value) for value in words) or "0"
    body = [
        f"static const int64_t LANES[] = {{{lanes}}};",
        "static int history[STEPS + 1][NEURONS];",
        "int main(void) {",
    ]
    for population in lowered.graph.populations:
        kernel = kernels[population.neuron_type]
        body.append(f"    {kernel}_state_t s_{population.name}[{population.n_neurons}];")
        body.append(
            f"    for (int i = 0; i < {population.n_neurons}; ++i) {kernel}_reset(&s_{population.name}[i]);"
        )
    body.append("    for (int t = 0; t < STEPS; ++t) {")
    body.append(f"        {word} current[NEURONS];")
    for population in lowered.graph.populations:
        base = offsets[population.name]
        for index in range(population.n_neurons):
            terms: list[str] = []
            for connection in qgraph.connections:
                if connection.dst != population.name:
                    continue
                weights = np.asarray(connection.weights, dtype=np.int64)
                source_base = offsets.get(connection.src)
                for column in range(weights.shape[1]):
                    weight = int(weights[index, column])
                    if weight == 0:
                        continue
                    if source_base is None:
                        lane = lane_offsets[connection.src] + column
                        terms.append(f"((LANES[{lane}] * {weight}) >> {fraction})")
                        continue
                    # The Studio lowering gives every projection one scalar delay.
                    delay = int(np.atleast_1d(np.asarray(connection.delay_steps))[0])
                    # The spike seen now was produced ``delay + 1`` steps ago.
                    terms.append(
                        f"(t - {delay + 1} >= 0 && history[t - {delay + 1}][{source_base + column}] "
                        f"? {weight} : 0)"
                    )
            acc = " + ".join(terms) if terms else "0"
            body.append(
                f"        current[{base + index}] = sat_word(wrap_acc((int64_t)0 + {acc}));"
            )
    for population in lowered.graph.populations:
        kernel = kernels[population.neuron_type]
        base = offsets[population.name]
        body.append(
            f"        for (int i = 0; i < {population.n_neurons}; ++i) history[t][{base} + i] = "
            f"{kernel}_step(&s_{population.name}[i], current[{base} + i]);"
        )
    body.extend(
        [
            "        for (int n = NEURONS - 1; n >= 0; --n) putchar(history[t][n] ? '1' : '0');",
            "        putchar('\\n');",
            "    }",
            "    return 0;",
            "}",
        ]
    )
    parts.append("\n".join(body) + "\n")
    return "\n".join(parts)


def _run_model(source: str, workdir: Path, timeout_seconds: float) -> tuple[str, ...]:
    model = workdir / "model"
    model.mkdir(parents=True)
    (model / "network.c").write_text(source, encoding="utf-8")
    binary = model / "network"
    subprocess.run(
        ["gcc", "-std=c11", "-O1", "-o", str(binary), str(model / "network.c")],
        check=True,
        capture_output=True,
        timeout=timeout_seconds,
    )
    output = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True, timeout=timeout_seconds
    ).stdout
    return tuple(output.splitlines())


def _studio_raster(graph: object, lowered: LoweredNetwork, steps: int) -> tuple[str, ...]:
    """Return the Studio's own run as rows in the hardware's spike-bus order."""
    result = simulate_graph(graph)
    neurons = lowered.graph.total_neurons
    raster = np.zeros((steps, neurons), dtype=np.int8)
    offset = 0
    offsets: dict[str, int] = {}
    for population in lowered.graph.populations:
        offsets[population.name] = offset
        offset += population.n_neurons
    for population in result["populations"]:
        base = offsets[population["id"]]
        for step, neuron in zip(population["events"]["step"], population["events"]["neuron"]):
            if step < steps:
                raster[step, base + neuron] = 1
    return tuple("".join("1" if bit else "0" for bit in row[::-1]) for row in raster)


__all__ = [
    "HARDWARE_MODULE_NAME",
    "NETWORK_COSIM_SCHEMA_VERSION",
    "HardwareCosimUnavailable",
    "NetworkCosim",
    "compile_lowered",
    "cosimulate",
]
