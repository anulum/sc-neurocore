# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — shared Python↔Verilog co-simulation primitives

"""Shared Python↔Verilog co-simulation primitives.

One responsibility: build a schema model (optionally pipelined), lower it to Verilog,
run it through ``iverilog`` + ``vvp``, and return the spike count. The co-simulation
test modules import these so each stays scoped to a single behaviour under test rather
than re-implementing the compile/simulate boilerplate.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from sc_neurocore.compiler.equation_compiler import generate_testbench
from sc_neurocore.neurons.universal_dsl import UniversalNeuron

HAS_IVERILOG = shutil.which("iverilog") is not None


def simulate(verilog: str, tb: str, module_name: str) -> int:
    """Compile RTL + testbench with iverilog, run vvp, return the reported spike count."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"
        rtl_path.write_text(verilog)
        tb_path.write_text(tb)
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")
        result = subprocess.run(["vvp", str(out_path)], capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")
        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def spike_count_method(model_name: str, n_steps: int, current: float, method: str) -> int:
    """Python golden spike count with an explicit integrator ``method`` override."""
    neuron = UniversalNeuron.from_schema(model_name, method_override=method)
    return sum(1 for _ in range(n_steps) if neuron.step(I=current))


def verilog_spike_count_method(
    model_name: str,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
    method: str,
) -> int:
    """Compile at ``method``/``(data_width, fraction)`` and simulate, returning spikes."""
    neuron = UniversalNeuron.from_schema(model_name, method_override=method)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_{method}_q{data_width - fraction}_{fraction}"

    verilog = neuron.to_verilog(module_name=module_name, data_width=data_width, fraction=fraction)
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=data_width,
        fraction=fraction,
    )
    return simulate(verilog, tb, module_name)


def verilog_spike_count_method_pipelined(
    model_name: str,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
    method: str,
    pipeline_stages: int,
) -> tuple[int, int]:
    """Compile pipelined at ``method``/Q-format, drive latency-aware, return ``(spikes, latency)``.

    The fill-counter FSM advances one logical step every ``latency + 1`` clocks and pulses
    ``spike_out`` only on the valid cycle, so the testbench runs ``latency + 1`` clocks per
    logical step; the spike count is then directly comparable to the combinational path.
    """
    neuron = UniversalNeuron.from_schema(model_name, method_override=method)
    eq_neuron = neuron.to_equation_neuron()
    module_name = (
        f"sc_{model_name}_{method}_pl{pipeline_stages}_q{data_width - fraction}_{fraction}"
    )

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
        pipeline_stages=pipeline_stages,
    )
    match = re.search(r"Pipeline latency: (\d+) cycle", verilog)
    latency = int(match.group(1)) if match else 0
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=data_width,
        fraction=fraction,
        cycles_per_step=latency + 1,
    )
    return simulate(verilog, tb, module_name), latency
