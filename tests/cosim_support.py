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

import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Mapping, cast

import numpy as np

from sc_neurocore.compiler.equation_compiler import Q88, generate_testbench
from sc_neurocore.compiler.verilog_compiler import compile_to_verilog as compile_to_verilog
from sc_neurocore.neurons.equation_builder import EquationNeuron
from sc_neurocore.neurons.models.connor_stevens import ConnorStevensNeuron
from sc_neurocore.neurons.models.dpi_neuron import DPINeuron
from sc_neurocore.neurons.models.fitzhugh_nagumo import FitzHughNagumoNeuron
from sc_neurocore.neurons.models.fitzhugh_rinzel import FitzHughRinzelNeuron
from sc_neurocore.neurons.models.glif import GLIFNeuron
from sc_neurocore.neurons.models.hodgkin_huxley import HodgkinHuxleyNeuron
from sc_neurocore.neurons.models.hindmarsh_rose import HindmarshRoseNeuron
from sc_neurocore.neurons.models.izhikevich2007 import Izhikevich2007Neuron
from sc_neurocore.neurons.models.mckean import McKeanNeuron
from sc_neurocore.neurons.models.mihalas_niebur import MihalasNieburNeuron
from sc_neurocore.neurons.models.morris_lecar import MorrisLecarNeuron
from sc_neurocore.neurons.models.perfect_integrator import PerfectIntegratorNeuron
from sc_neurocore.neurons.models.pernarowski import PernarowskiNeuron
from sc_neurocore.neurons.models.terman_wang import TermanWangOscillator
from sc_neurocore.neurons.models.wang_buzsaki import WangBuzsakiNeuron
from sc_neurocore.neurons.models.wilson_hr import WilsonHRNeuron
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


def _python_spike_count(model_name: str, n_steps: int, current: float) -> int:
    """Run a model in Python and return the spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    spikes = 0
    for _ in range(n_steps):
        if neuron.step(I=current):
            spikes += 1
    return spikes


def _verilog_spike_count(model_name: str, n_steps: int, current: float) -> int:
    """Compile a model to Verilog, simulate with iverilog, return spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}"

    verilog = neuron.to_verilog(module_name=module_name)
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"

        rtl_path.write_text(verilog)
        tb_path.write_text(tb)

        # Compile
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{result.stderr}")

        # Simulate
        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        # Parse spike count from output: "Simulation complete: N spikes in M cycles"
        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _perfect_integrator_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored perfect-integrator spike count for comparison."""
    neuron = PerfectIntegratorNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _izhikevich2007_hand_euler_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Izhikevich 2007 (Euler) spike count for comparison."""
    neuron = Izhikevich2007Neuron(integrator="euler")
    return sum(neuron.step(current) for _ in range(n_steps))


def _dpi_neuron_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored DPI (current-mode Euler) spike count for comparison."""
    neuron = DPINeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _fitzhugh_nagumo_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored FitzHugh-Nagumo (RK4, rising-edge crossing) spike count."""
    neuron = FitzHughNagumoNeuron(
        dt=0.1, v=-1.0, w=-0.5, a=0.7, b=0.8, epsilon=0.08, v_threshold=1.0
    )
    return sum(neuron.step(current) for _ in range(n_steps))


def _fitzhugh_rinzel_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored FitzHugh-Rinzel RK4 upward-crossing count."""
    neuron = FitzHughRinzelNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _hindmarsh_rose_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Hindmarsh-Rose RK4 upward-crossing count."""
    neuron = HindmarshRoseNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _glif_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored GLIF candidate-first RK4 spike count."""
    neuron = GLIFNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _pernarowski_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Pernarowski RK4 upward-crossing count."""
    neuron = PernarowskiNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _terman_wang_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Terman-Wang RK4 upward-crossing count."""
    neuron = TermanWangOscillator()
    return sum(neuron.step(current) for _ in range(n_steps))


def _wilson_hr_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Wilson-HR RK4 hard-reset spike count."""
    neuron = WilsonHRNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


# Sustained relaxation-oscillation operating point mirrored by the bundled ``mckean`` schema.
# The default hand-model regime (epsilon=0.01) is a single-transient knife-edge; epsilon=0.2 /
# gamma=0.5 puts the piecewise-linear caricature on a robust limit cycle whose upward v_peak
# crossings survive Q16.16 rounding, so the min/max RK4 datapath co-simulates bit-exactly.
_MCKEAN_PARAMS = {"a": 0.25, "epsilon": 0.2, "gamma": 0.5, "v_peak": 0.8}


def _mckean_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored McKean (RK4, rising-edge crossing) spike count."""
    neuron = McKeanNeuron(dt=0.1, v=0.0, w=0.0, **_MCKEAN_PARAMS)
    return sum(neuron.step(current) for _ in range(n_steps))


def _morris_lecar_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Morris-Lecar (RK4, rising-edge crossing) spike count.

    The bundled ``morris_lecar`` schema mirrors ``MorrisLecarNeuron``'s maintained
    defaults exactly (RK4 integrator, no reset, ``v >= v_threshold`` upward crossing,
    ``phi = 1/15``), so the default construction is the enrolled operating point.
    """
    neuron = MorrisLecarNeuron()
    return sum(neuron.step(current) for _ in range(n_steps))


def _connor_stevens_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Connor-Stevens macro-step (RK4, crossing) spike count.

    The maintained ``ConnorStevensNeuron.step`` is a 1 ms macro step of 100 inner
    four-stage RK4 sub-steps (``dt=0.01``) with a rising-edge ``v >= v_threshold`` crossing
    on the macro boundary and no reset. The bundled ``connor_stevens`` schema mirrors this
    exactly (``method="rk4"``, ``substeps=100``, ``detection="crossing"``), so one hand
    ``step()`` corresponds to one schema macro ``step()``.
    """
    neuron = ConnorStevensNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _hodgkin_huxley_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Hodgkin-Huxley macro-step (RK4, crossing) spike count.

    ``HodgkinHuxleyNeuron.step`` is a 1 ms macro step of 100 inner sub-steps (``dt=0.01``)
    with a rising-edge ``v >= v_threshold`` crossing on the macro boundary and no reset. The
    bundled ``hodgkin_huxley`` schema mirrors the ``integrator="rk4"`` path exactly
    (``method="rk4"``, ``substeps=100``, ``detection="crossing"``) — the simultaneous RK4,
    not the Gauss-Seidel default ``baseline_euler`` — so one hand ``step()`` corresponds to
    one schema macro ``step()``.
    """
    neuron = HodgkinHuxleyNeuron(integrator="rk4")
    return sum(neuron.step(current) for _ in range(n_macro_steps))


def _wang_buzsaki_hand_spike_count(n_macro_steps: int, current: float) -> int:
    """Return the hand-authored Wang-Buzsaki macro-step (Gauss-Seidel, crossing) spike count.

    ``WangBuzsakiNeuron.step`` is a 0.5 ms macro step of 50 inner sub-steps (``dt=0.01``)
    advanced sequentially (the gating variables ``h``/``n`` from the old voltage, then the
    membrane voltage ``v`` from the new gates), with a rising-edge ``v >= v_threshold``
    crossing on the macro boundary and no reset. The bundled ``wang_buzsaki`` schema mirrors
    that path exactly (``method="gauss_seidel"``, ``substeps=50``, state ordered ``h, n, v``,
    ``detection="crossing"``), so one hand ``step()`` corresponds to one schema macro
    ``step()``. The neuron is constructed once so the state accumulates across the train.
    """
    neuron = WangBuzsakiNeuron()
    return sum(neuron.step(current) for _ in range(n_macro_steps))


# Adaptive-threshold operating point mirrored by the bundled ``mihalas_niebur`` schema.
# ``theta_reset`` (1.3) exceeds ``theta_inf`` (1.0), so the max() threshold floor engages
# on every spike and the fractional taus/coefficients stress the fixed-point datapath.
_MIHALAS_NIEBUR_PARAMS = {
    "v_rest": 0.0,
    "v_reset": 0.0,
    "theta_reset": 1.3,
    "theta_inf": 1.0,
    "tau_v": 10.0,
    "tau_theta": 40.0,
    "tau_1": 15.0,
    "tau_2": 80.0,
    "a": 0.1,
    "b": 0.1,
    "r1": 0.2,
    "r2": -0.15,
}


def _mihalas_niebur_hand_spike_count(n_steps: int, current: float) -> int:
    """Return the hand-authored Mihalas-Niebur (RK4) spike count for comparison."""
    neuron = MihalasNieburNeuron(dt=1.0, **_MIHALAS_NIEBUR_PARAMS)
    return sum(neuron.step(current) for _ in range(n_steps))


def _lif_schema_precision_values() -> dict[str, float]:
    """Return LIF schema values checked by the public precision CLI."""
    schema = UniversalNeuron.from_schema("lif").schema
    parameters = cast(Mapping[str, float], schema.get("parameters", {}))
    state = cast(Mapping[str, float], schema.get("state", {}))
    return {**parameters, **state}


def _verilog_spike_count_q412(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q4.12 precision and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q412"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=16,
        fraction=12,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=16,
        fraction=12,
    )

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

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _verilog_spike_count_q1616(model_name: str, n_steps: int, current: float) -> int:
    """Compile at Q16.16 precision (32-bit) and simulate, returning spike count."""
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    module_name = f"sc_{model_name}_q1616"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=32,
        fraction=16,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )

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

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _rulkov_map_verilog_q1616_trace(n_steps: int, current: float) -> list[tuple[int, float, float]]:
    """Return the emitted Rulkov RTL's committed Q16.16 state trace.

    The testbench samples the generated module's synchronous ``x_reg`` and
    ``y_reg`` state after each active clock edge. These registers are the map
    recurrence itself; the public state outputs retain the pre-threshold value
    on a spiking cycle, so sampling the committed registers
    avoids confusing that interface convention with the next-state trajectory.
    """
    neuron = UniversalNeuron.from_schema("rulkov_map")
    module_name = "sc_rulkov_map_q1616_trace"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_sc_rulkov_map_q1616_trace;",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] x_out;",
            "wire signed [31:0] y_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .x_out(x_out), .y_out(y_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("RULKOV_TRACE %0d %0d %0d", spike_out, uut.x_reg, uut.y_reg);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        tb_path = root / f"tb_{module_name}.v"
        out_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

    scale = float(1 << 16)
    rows = re.findall(r"^RULKOV_TRACE (-?\d+) (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    trace = [(int(spike), int(x_q) / scale, int(y_q) / scale) for spike, x_q, y_q in rows]
    assert len(trace) == n_steps, (
        f"Rulkov RTL emitted {len(trace)} trace rows; expected {n_steps}:\n{simulation.stdout}"
    )
    return trace


def _ibarz_tanaka_verilog_q1616_trace(
    n_steps: int, current: float
) -> list[tuple[int, float, float]]:
    """Return the emitted Ibarz-Tanaka RTL's committed Q16.16 trace."""
    neuron = UniversalNeuron.from_schema("ibarz_tanaka_map")
    module_name = "sc_ibarz_tanaka_rulkov_map"
    verilog = neuron.to_verilog(module_name=module_name, data_width=32, fraction=16)
    current_q = Q88(data_width=32, fraction=16).encode(current)
    testbench = "\n".join(
        [
            "`timescale 1ns / 1ps",
            f"module tb_{module_name};",
            "reg clk = 1'b0;",
            "reg rst_n = 1'b0;",
            "wire spike_out;",
            "wire signed [31:0] v_out;",
            "wire signed [31:0] u_out;",
            "always #5 clk = ~clk;",
            f"{module_name} uut (",
            "    .clk(clk), .rst_n(rst_n),",
            f"    .I_t(32'sd{current_q}),",
            "    .spike_out(spike_out), .v_out(v_out), .u_out(u_out)",
            ");",
            "integer step_index;",
            "initial begin",
            "    #23; rst_n = 1'b1;",
            f"    for (step_index = 0; step_index < {n_steps}; step_index = step_index + 1) begin",
            "        @(posedge clk); #1;",
            '        $display("IBARZ_TRACE %0d %0d %0d", spike_out, uut.v_reg, uut.u_reg);',
            "    end",
            "    $finish;",
            "end",
            "endmodule",
        ]
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        rtl_path = root / f"{module_name}.v"
        tb_path = root / f"tb_{module_name}.v"
        out_path = root / f"tb_{module_name}"
        rtl_path.write_text(verilog, encoding="utf-8")
        tb_path.write_text(testbench, encoding="utf-8")
        subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )
        simulation = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        )

    scale = float(1 << 16)
    rows = re.findall(r"^IBARZ_TRACE (-?\d+) (-?\d+) (-?\d+)$", simulation.stdout, re.MULTILINE)
    trace = [(int(event), int(v_q) / scale, int(u_q) / scale) for event, v_q, u_q in rows]
    assert len(trace) == n_steps, (
        f"Ibarz-Tanaka RTL emitted {len(trace)} trace rows; expected {n_steps}:\n"
        f"{simulation.stdout}"
    )
    return trace


def _neuron_verilog_spike_count_q1616(
    neuron: EquationNeuron, n_steps: int, current: float, module_name: str
) -> int:
    """Compile a raw ``EquationNeuron`` to Q16.16 RTL, simulate, return the spike count.

    Unlike :func:`_verilog_spike_count_q1616` this takes a constructed neuron directly (not a
    bundled schema name), so it can co-simulate an in-test configuration such as an artificial
    sub-step count on a polynomial oscillator.
    """
    verilog = compile_to_verilog(neuron, module_name=module_name, data_width=32, fraction=16)
    tb = generate_testbench(
        neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=32,
        fraction=16,
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        tb_path = Path(tmpdir) / f"tb_{module_name}.v"
        out_path = Path(tmpdir) / f"tb_{module_name}"
        rtl_path.write_text(verilog)
        tb_path.write_text(tb)
        compile_result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if compile_result.returncode != 0:
            raise RuntimeError(f"iverilog compile failed:\n{compile_result.stderr}")
        run_result = subprocess.run(
            ["vvp", str(out_path)], capture_output=True, text=True, timeout=60
        )
        if run_result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{run_result.stderr}")
        match = re.search(r"(\d+) spikes", run_result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{run_result.stdout}")
        return int(match.group(1))


def _fitzhugh_nagumo_substep_neuron(substeps: int) -> EquationNeuron:
    """Build the faithful FitzHugh-Nagumo oscillator with an artificial sub-step count.

    FitzHugh-Nagumo is polynomial, so its Q16.16 datapath is bit-exact against float64; giving
    it ``substeps`` inner steps lets the macro-step lowering be validated on a model whose only
    residual would be a logic error (no look-up-table quantisation to confound the comparison).
    """
    return EquationNeuron(
        equations={
            "v": "v - v * v * v / 3.0 - w + I",
            "w": "epsilon * (v + a - b * w)",
        },
        parameters={"a": 0.7, "b": 0.8, "epsilon": 0.08, "v_threshold": 1.0},
        state={"v": -1.0, "w": -0.5},
        threshold="v >= v_threshold",
        dt=0.1,
        method="rk4",
        detection="crossing",
        substeps=substeps,
    )


def _verilog_spike_count_generic(
    model_name: str,
    n_steps: int,
    current: float,
    data_width: int,
    fraction: int,
) -> int:
    """Compile at arbitrary (data_width, fraction) and simulate, returning spike count.

    This is the universal co-simulation helper — all precision-specific
    helpers (_verilog_spike_count, _verilog_spike_count_q412, etc.) are
    special cases of this function.
    """
    neuron = UniversalNeuron.from_schema(model_name)
    eq_neuron = neuron.to_equation_neuron()
    mode_tag = f"q{data_width - fraction}_{fraction}"
    module_name = f"sc_{model_name}_{mode_tag}"

    verilog = neuron.to_verilog(
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )
    tb = generate_testbench(
        eq_neuron,
        module_name=module_name,
        n_steps=n_steps,
        input_current=current,
        data_width=data_width,
        fraction=fraction,
    )

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

        result = subprocess.run(
            ["vvp", str(out_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

        match = re.search(r"(\d+) spikes", result.stdout)
        if not match:
            raise RuntimeError(f"Could not parse spike count from:\n{result.stdout}")
        return int(match.group(1))


def _verilog_compiles(model_name: str) -> bool:
    """Return whether a model's generated Verilog is accepted by iverilog."""
    neuron = UniversalNeuron.from_schema(model_name)
    module_name = f"sc_{model_name}"
    verilog = neuron.to_verilog(module_name=module_name)
    with tempfile.TemporaryDirectory() as tmpdir:
        rtl_path = Path(tmpdir) / f"{module_name}.v"
        out_path = Path(tmpdir) / f"{module_name}.out"
        rtl_path.write_text(verilog)
        result = subprocess.run(
            ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0


def _summarise(recorded: dict[str, list[float]], spikes: list[int]) -> dict[str, float]:
    """Return the shared spike-count / first-spike-step / per-variable feature map.

    Every reference helper that tracks a per-step ``spikes`` list and one or more
    recorded state-variable trajectories reduces them to the same feature contract: a
    total spike count, the 1-indexed first-spike step (``-1`` when silent), and the
    final / minimum / maximum / mean of each recorded variable. Centralising the tail
    keeps the independent-parity helpers byte-identical in how they summarise, so a
    drift in one helper's reduction cannot silently diverge from the others.

    Parameters
    ----------
    recorded:
        Mapping from state-variable name to its per-step trajectory.
    spikes:
        Per-step spike indicators (``1`` on a spiking step, ``0`` otherwise).

    Returns
    -------
    dict of str to float
        The feature map keyed by ``spike_count``, ``first_spike_step``, and
        ``final.<var>`` / ``min.<var>`` / ``max.<var>`` / ``mean.<var>`` per variable.
    """
    features: dict[str, float] = {
        "spike_count": float(math.fsum(spikes)),
        "first_spike_step": float(
            next((index for index, spike in enumerate(spikes, start=1) if spike), -1)
        ),
    }
    for variable, values in recorded.items():
        features[f"final.{variable}"] = values[-1]
        features[f"min.{variable}"] = min(values)
        features[f"max.{variable}"] = max(values)
        features[f"mean.{variable}"] = math.fsum(values) / len(values)
    return features


def _closed_form_features(
    *,
    initial: float,
    steady: float,
    tau: float,
    dt: float,
    steps: int,
) -> dict[str, float]:
    values = [
        steady + (initial - steady) * math.exp(-(step * dt) / tau) for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _quadratic_if_zero_current_features(*, dt: float, steps: int) -> dict[str, float]:
    values = [-1.0 / (1.0 + step * dt) for step in range(1, steps + 1)]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.v": values[-1],
        "min.v": min(values),
        "max.v": max(values),
        "mean.v": math.fsum(values) / len(values),
    }


def _perfect_integrator_sawtooth_features(
    *,
    current: float,
    dt: float,
    steps: int,
    c_m: float = 1.0,
    v_threshold: float = 1.0,
    v_reset: float = 0.0,
) -> dict[str, float]:
    """Return exact post-reset features for constant-current perfect integration."""
    values: list[float] = []
    spikes: list[int] = []
    voltage = v_reset
    increment = current * dt / c_m
    for _ in range(steps):
        voltage += increment
        if voltage >= v_threshold:
            spikes.append(1)
            voltage = v_reset
        else:
            spikes.append(0)
        values.append(voltage)

    return _summarise({"v": values}, spikes)


def _theta_constant_current_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return continuous theta-neuron phase features for constant positive current."""
    if current <= 0.0:
        msg = "theta analytic helper requires positive current"
        raise ValueError(msg)
    root_current = math.sqrt(current)
    values = [
        2.0 * math.atan(root_current * math.tan(root_current * step * dt))
        for step in range(1, steps + 1)
    ]
    return {
        "spike_count": 0.0,
        "first_spike_step": -1.0,
        "final.theta": values[-1],
        "min.theta": min(values),
        "max.theta": max(values),
        "mean.theta": math.fsum(values) / len(values),
    }


def _resonate_fire_linear_euler_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact Euler features for the linear resonate-and-fire schema."""
    omega = 0.5
    damping = -0.1
    threshold = 1.0
    x = 0.0
    y = 0.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dx = damping * x - omega * y + current
        dy = omega * x + damping * y
        x_next = x + dt * dx
        y_next = y + dt * dy
        if x_next > threshold:
            spikes.append(1)
            x = 0.0
            y = 0.0
        else:
            spikes.append(0)
            x = x_next
            y = y_next
        x_values.append(x)
        y_values.append(y)

    return _summarise({"x": x_values, "y": y_values}, spikes)


def _glif_driven_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven GLIF5 flow and adaptive reset.

    The maintained Allen Institute GLIF5 model advances four coupled linear states:
    the membrane potential, adaptive threshold, and two after-spike currents. This
    independent recurrence evaluates all four classical RK4 stages from the same
    pre-step state, then applies the candidate-level ``v >= theta`` decision and the
    candidate-first voltage, threshold, and current reset increments. A driven tonic
    train therefore exercises both the continuous flow and every reset surface rather
    than validating only a silent linear tail.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``theta``, ``i_asc1``, and ``i_asc2``
        state variables plus spike-count and first-spike-step features.
    """
    theta_inf = -50.0
    v_rest = -70.0
    v_reset = -70.0
    tau_m = 10.0
    tau_theta = 100.0
    tau_asc1 = 10.0
    tau_asc2 = 200.0
    a_theta = 0.01
    delta_theta = 2.0
    r_asc1 = 1.0
    r_asc2 = 0.5
    resistance = 1.0

    v = v_rest
    theta = theta_inf
    i_asc1 = 0.0
    i_asc2 = 0.0
    half_dt = 0.5 * dt
    recorded: dict[str, list[float]] = {"v": [], "theta": [], "i_asc1": [], "i_asc2": []}
    spikes: list[int] = []

    def derivatives(
        membrane: float,
        threshold: float,
        asc1: float,
        asc2: float,
    ) -> tuple[float, float, float, float]:
        return (
            (-(membrane - v_rest) + resistance * current + asc1 + asc2) / tau_m,
            (theta_inf - threshold + a_theta * (membrane - v_rest)) / tau_theta,
            -asc1 / tau_asc1,
            -asc2 / tau_asc2,
        )

    for _ in range(steps):
        k1 = derivatives(v, theta, i_asc1, i_asc2)
        k2 = derivatives(
            v + half_dt * k1[0],
            theta + half_dt * k1[1],
            i_asc1 + half_dt * k1[2],
            i_asc2 + half_dt * k1[3],
        )
        k3 = derivatives(
            v + half_dt * k2[0],
            theta + half_dt * k2[1],
            i_asc1 + half_dt * k2[2],
            i_asc2 + half_dt * k2[3],
        )
        k4 = derivatives(
            v + dt * k3[0],
            theta + dt * k3[1],
            i_asc1 + dt * k3[2],
            i_asc2 + dt * k3[3],
        )
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        i_asc1 = i_asc1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i_asc2 = i_asc2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if v >= theta:
            spikes.append(1)
            v = v_reset
            theta = theta + delta_theta
            i_asc1 = i_asc1 + r_asc1
            i_asc2 = i_asc2 + r_asc2
        else:
            spikes.append(0)
        recorded["v"].append(v)
        recorded["theta"].append(theta)
        recorded["i_asc1"].append(i_asc1)
        recorded["i_asc2"].append(i_asc2)

    return _summarise(recorded, spikes)


def _izhikevich_rs_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the regular-spiking Izhikevich recurrence.

    The Izhikevich (2003) quadratic membrane and linear recovery equations are
    advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane crosses the ``v > 30`` peak. The reference is therefore an independent
    re-derivation of the committed spike-bearing trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.02
    b = 0.2
    c = -65.0
    d = 8.0
    v = -65.0
    u = -14.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = 0.04 * v**2 + 5 * v + 140 - u + current
        du = a * (b * v - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next > 30:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)


def _izhikevich2007_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the Izhikevich 2007 recurrence.

    The Izhikevich (2007) biophysical quadratic membrane ``C dv/dt =
    k (v - vr) (v - vt) - u + I`` and linear recovery ``du/dt = a (b (v - vr) - u)``
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies, and the ``v = c``, ``u = u + d`` reset fires whenever the post-update
    membrane reaches the ``v >= vpeak`` peak. The right-hand side is polynomial, so
    the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed regular-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``u`` state variables plus
        spike-count and first-spike-step features.
    """
    c_m = 100.0
    k = 0.7
    vr = -60.0
    vt = -40.0
    vpeak = 35.0
    a = 0.03
    b = -2.0
    c = -50.0
    d = 100.0
    v = -60.0
    u = 0.0
    v_values: list[float] = []
    u_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (k * (v - vr) * (v - vt) - u + current) / c_m
        du = a * (b * (v - vr) - u)
        v_next = v + dv * dt
        u_next = u + du * dt
        if v_next >= vpeak:
            spikes.append(1)
            v_next = c
            u_next = u_next + d
        else:
            spikes.append(0)
        v, u = v_next, u_next
        v_values.append(v)
        u_values.append(u)

    return _summarise({"v": v_values, "u": u_values}, spikes)


def _dpi_neuron_driven_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the driven DPI current-mode recurrence.

    The DYNAP-SE differential-pair-integrator membrane ``tau dI_mem/dt =
    -I_mem + gain * I_syn + I_leak`` (Chicca et al. 2014) is advanced with the same
    explicit-Euler update the schema runner applies, and the ``i_mem = i_reset`` reset
    fires whenever the post-update current reaches the ``i_mem >= i_threshold`` level.
    The right-hand side is linear, so the recurrence reproduces the schema runner
    bit-for-bit — an independent re-derivation of the committed driven-spiking trace,
    not a copy of the runner. The non-negative drive keeps ``i_mem`` non-negative, so
    the source model's ``max(i_mem, 0)`` current rectification is inert and correctly
    absent from this continuous update.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``i_mem`` state variable plus spike-count and
        first-spike-step features.
    """
    i_threshold = 1.0
    i_reset = 0.0
    i_leak = 0.01
    tau = 20.0
    gain = 1.0
    i_mem = 0.0
    i_mem_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        di = (-i_mem + gain * current + i_leak) / tau
        i_mem_next = i_mem + di * dt
        if i_mem_next >= i_threshold:
            spikes.append(1)
            i_mem_next = i_reset
        else:
            spikes.append(0)
        i_mem = i_mem_next
        i_mem_values.append(i_mem)

    return _summarise({"i_mem": i_mem_values}, spikes)


def _mihalas_niebur_driven_rk4_features(
    *, current: float, dt: float, steps: int
) -> dict[str, float]:
    """Return exact fourth-order Runge-Kutta features for the driven Mihalas-Niebur flow.

    The generalised integrate-and-fire flow (Mihalaş & Niebur 2009) advances four linear
    states — membrane ``dv/dt = (-(v - v_rest) + i1 + i2 + I) / tau_v``, adaptive threshold
    ``dtheta/dt = (theta_inf - theta + a (v - v_rest)) / tau_theta`` and two spike-triggered
    currents ``di1/dt = -i1 / tau_1``, ``di2/dt = -i2 / tau_2`` — with the classical RK4
    step the schema runner applies, and the adaptive reset ``v = v_reset + b (v - v_rest)``,
    ``theta = max(theta, theta_reset)``, ``i1 += r1``, ``i2 += r2`` fires whenever the
    post-step membrane reaches the state-to-state ``v >= theta`` threshold. Every derivative
    is linear, so the recurrence reproduces the schema runner bit-for-bit — an independent
    re-derivation of the committed driven-spiking trace, not a copy of the runner. Because
    ``theta_reset`` (1.3) exceeds ``theta_inf`` (1.0) the max() threshold floor engages on
    every spike, so the state-to-state comparison is a genuine adaptive threshold.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``theta``, ``i1`` and ``i2`` state variables
        plus spike-count and first-spike-step features.
    """
    v_rest = 0.0
    v_reset = 0.0
    theta_reset = 1.3
    theta_inf = 1.0
    tau_v = 10.0
    tau_theta = 40.0
    tau_1 = 15.0
    tau_2 = 80.0
    a = 0.1
    b = 0.1
    r1 = 0.2
    r2 = -0.15
    v = 0.0
    theta = 1.0
    i1 = 0.0
    i2 = 0.0
    half_dt = 0.5 * dt
    v_values: list[float] = []
    theta_values: list[float] = []
    i1_values: list[float] = []
    i2_values: list[float] = []
    spikes: list[int] = []

    def deriv(vv: float, th: float, j1: float, j2: float) -> tuple[float, float, float, float]:
        return (
            (-(vv - v_rest) + j1 + j2 + current) / tau_v,
            (theta_inf - th + a * (vv - v_rest)) / tau_theta,
            -j1 / tau_1,
            -j2 / tau_2,
        )

    for _ in range(steps):
        k1 = deriv(v, theta, i1, i2)
        k2 = deriv(
            v + half_dt * k1[0],
            theta + half_dt * k1[1],
            i1 + half_dt * k1[2],
            i2 + half_dt * k1[3],
        )
        k3 = deriv(
            v + half_dt * k2[0],
            theta + half_dt * k2[1],
            i1 + half_dt * k2[2],
            i2 + half_dt * k2[3],
        )
        k4 = deriv(
            v + dt * k3[0],
            theta + dt * k3[1],
            i1 + dt * k3[2],
            i2 + dt * k3[3],
        )
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        theta = theta + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        i1 = i1 + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        i2 = i2 + dt * (k1[3] + 2.0 * k2[3] + 2.0 * k3[3] + k4[3]) / 6.0
        if v >= theta:
            spikes.append(1)
            v = v_reset + b * (v - v_rest)
            theta = max(theta, theta_reset)
            i1 = i1 + r1
            i2 = i2 + r2
        else:
            spikes.append(0)
        v_values.append(v)
        theta_values.append(theta)
        i1_values.append(i1)
        i2_values.append(i2)

    return _summarise(
        {"v": v_values, "theta": theta_values, "i1": i1_values, "i2": i2_values}, spikes
    )


def _fitzhugh_nagumo_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven FitzHugh-Nagumo oscillator.

    The FitzHugh (1961) cubic membrane and linear recovery equations are advanced
    with the same four-stage RK4 step and rising-edge spike detection the faithful
    schema runner applies, with **no reset** — the re-enrolled model is a genuine
    relaxation oscillator whose spikes are upward ``v >= 1`` threshold crossings, not
    integrate-and-fire resets. The cube is written ``v * v * v`` (not ``v ** 3``) so
    it is the exact IEEE multiplication the runner and the hand model evaluate. The
    reference is an independent re-derivation of the committed relaxation-oscillation
    trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.7
    b = 0.8
    epsilon = 0.08
    threshold = 1.0
    v = -1.0
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + current,
            epsilon * (v_state + a - b * w_state),
        )

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _fitzhugh_rinzel_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven FitzHugh-Rinzel flow.

    The Rinzel (1987) three-state qualitative burster extends the FitzHugh-Nagumo
    fast subsystem with the ultra-slow ``y`` modulation equation. This independent
    recurrence advances all three coupled equations with one simultaneous four-stage
    RK4 step, then applies the maintained rising-edge ``v >= 1`` crossing decision
    without resetting any state. The cube is written ``v * v * v`` to reproduce the
    exact IEEE operation order of the hand model and schema runner; the recurrence is
    re-derived here rather than calling either implementation.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v``, ``w``, and ``y``, plus the spike count and
        first-spike step.
    """
    a = 0.7
    b = 0.8
    c = -0.775
    d = 1.0
    delta = 0.08
    mu = 0.0001
    threshold = 1.0
    v = -1.0
    w = -0.5
    y = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, y_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state + y_state + current,
            delta * (a + v_state - b * w_state),
            mu * (c - v_state - d * y_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, y)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            y + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            y + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], y + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        y = y + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        y_values.append(y)

    return _summarise({"v": v_values, "w": w_values, "y": y_values}, spikes)


def _pernarowski_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the autonomous Pernarowski flow.

    The Pernarowski (1994) beta-cell model couples a fast cubic coordinate to
    recovery ``w`` and ultra-slow adaptation ``z``. This independent recurrence
    advances all three equations simultaneously with classical four-stage RK4,
    then applies the maintained rising-edge ``v >= 0.5`` crossing decision
    without resetting state. It is re-derived here rather than calling the hand
    model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v``, ``w``, and ``z``, plus the spike count
        and first-spike step.
    """
    alpha = 0.1
    beta = 0.5
    eps1 = 0.1
    eps2 = 0.001
    gamma = 0.5
    threshold = 0.5
    v = -1.0
    w = 0.0
    z = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float, z_state: float) -> tuple[float, float, float]:
        return (
            v_state - v_state * v_state * v_state / 3.0 - w_state - z_state + current,
            eps1 * (v_state - gamma * w_state + alpha),
            eps2 * (beta * (v_state + 0.7) - z_state),
        )

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w, z)
        k2 = deriv(
            v + 0.5 * dt * k1[0],
            w + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
        )
        k3 = deriv(
            v + 0.5 * dt * k2[0],
            w + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
        )
        k4 = deriv(v + dt * k3[0], w + dt * k3[1], z + dt * k3[2])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        z = z + dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)
        z_values.append(z)

    return _summarise({"v": v_values, "w": w_values, "z": z_values}, spikes)


def _terman_wang_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Terman-Wang LEGION oscillator.

    This independent recurrence re-derives the maintained two-state Terman-Wang
    (1995) cubic fast nullcline and ``tanh``-gated slow recovery equation. It
    advances both states simultaneously through four Runge-Kutta stages, then
    applies the no-reset rising-edge ``v >= 1.5`` crossing decision without
    calling the hand model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for ``v`` and ``w``, plus the spike count and
        first-spike step.
    """
    alpha = 3.0
    beta = 0.2
    epsilon = 0.02
    rho = 0.0
    threshold = 1.5
    v = -1.5
    w = -0.5
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        fast = 3.0 * v_state - v_state * v_state * v_state + 2.0
        recovery = alpha * (1.0 + math.tanh(v_state / beta))
        return fast - w_state + current + rho, epsilon * (recovery - w_state)

    for _ in range(steps):
        v_prev = v
        k1 = deriv(v, w)
        k2 = deriv(v + 0.5 * dt * k1[0], w + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], w + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], w + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        w = w + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spikes.append(1 if (v >= threshold and v_prev < threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _wilson_hr_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the Wilson-HR cortical model.

    This independent recurrence re-derives Wilson's two-state polynomial flow,
    advances ``v`` and ``r`` simultaneously through four Runge-Kutta stages, and
    applies the level ``v >= 0.4`` spike decision. A spike hard-resets only ``v``
    to ``-0.7``; the RK4 candidate recovery state is preserved. The helper does not
    call the hand model or schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference features for post-reset ``v`` and candidate ``r``, plus the
        spike count and first-spike step.
    """
    tau_r = 1.9
    threshold = 0.4
    reset_voltage = -0.7
    v = -0.7
    r = 0.1
    v_values: list[float] = []
    r_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, r_state: float) -> tuple[float, float]:
        membrane = -(17.81 + 47.71 * v_state + 32.63 * v_state * v_state) * (v_state - 0.55)
        recovery_coupling = -26.0 * r_state * (v_state + 0.92)
        return (
            membrane + recovery_coupling + current,
            (-r_state + 1.35 * v_state + 1.03) / tau_r,
        )

    for _ in range(steps):
        k1 = deriv(v, r)
        k2 = deriv(v + 0.5 * dt * k1[0], r + 0.5 * dt * k1[1])
        k3 = deriv(v + 0.5 * dt * k2[0], r + 0.5 * dt * k2[1])
        k4 = deriv(v + dt * k3[0], r + dt * k3[1])
        v = v + dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        r = r + dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        spike = int(v >= threshold)
        if spike:
            v = reset_voltage
        spikes.append(spike)
        v_values.append(v)
        r_values.append(r)

    return _summarise({"v": v_values, "r": r_values}, spikes)


def _mckean_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven McKean oscillator.

    The McKean (1970) piecewise-linear FitzHugh-Nagumo caricature replaces the cubic
    membrane nullcline with the three-branch function ``f(v) = min(max(-v, v - a),
    1 - v)`` (min/max are supported by the schema DSL). The membrane and linear
    recovery equations are advanced with the same four-stage RK4 step and rising-edge
    ``v >= v_peak`` crossing detection the faithful schema runner applies, with **no
    reset** — the enrolled operating point (``epsilon = 0.2``, ``gamma = 0.5``,
    ``I = 0.6``) is a sustained relaxation oscillator whose spikes are upward threshold
    crossings. The right-hand side is exact arithmetic (comparisons and linear pieces,
    no cube or transcendental), so the reference is an independent re-derivation of the
    committed trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    a = 0.25
    epsilon = 0.2
    gamma = 0.5
    v_peak = 0.8
    v = 0.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        f_v = min(max(-v_state, v_state - a), 1.0 - v_state)
        return f_v - w_state + current, epsilon * (v_state - gamma * w_state)

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        spikes.append(1 if (v >= v_peak and v_prev < v_peak) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _adex_subthreshold_euler_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact explicit-Euler features for the subthreshold AdEx recurrence.

    The Brette-Gerstner (2005) exponential membrane and linear adaptation equations
    are advanced with the same simultaneous explicit-Euler update the schema runner
    applies. For the resting zero-current protocol the ``v > -50`` threshold is never
    reached, so the ``v = v_reset``, ``w = w + b`` reset stays inactive and the
    reference is an independent re-derivation of the committed quiet trajectory.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    v_rest = -65.0
    v_reset = -68.0
    v_rh = -55.0
    delta_t = 2.0
    tau = 20.0
    tau_w = 100.0
    a = 0.5
    b_adapt = 7.0
    capacitance = 200.0
    v = -65.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        dv = (-(v - v_rest) + delta_t * math.exp((v - v_rh) / delta_t)) / tau + (
            -w + current
        ) / capacitance
        dw = (a * (v - v_rest) - w) / tau_w
        v_next = v + dv * dt
        w_next = w + dw * dt
        if v_next > -50:
            spikes.append(1)
            v_next = v_reset
            w_next = w_next + b_adapt
        else:
            spikes.append(0)
        v, w = v_next, w_next
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _exp_if_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return independent RK4 features for the source-bound driven EIF recurrence.

    Fourcaud-Trocmé et al. (2003), Equations 6 and 10, define the leak plus
    exponential current. This re-derivation uses the fitted ``V_T``, slope,
    leak, reset and the paper's ``+30 mV`` finite simulation cutoff. RK4 stages
    are bounded at that event surface, matching the maintained deterministic
    recurrence without importing the hand model or the schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for voltage, spike count, and first-spike step.
    """
    v_rest = -65.0
    v_reset = -68.0
    v_threshold = 30.0
    v_rh = -59.9
    delta_t = 3.48
    tau = 10.0
    v = -65.0
    v_values: list[float] = []
    spikes: list[int] = []

    def rhs(stage_v: float) -> float:
        bounded_v = min(stage_v, v_threshold)
        return (
            -(bounded_v - v_rest) + delta_t * math.exp((bounded_v - v_rh) / delta_t) + current
        ) / tau

    for _ in range(steps):
        k1 = rhs(v)
        k2 = rhs(v + 0.5 * dt * k1)
        k3 = rhs(v + 0.5 * dt * k2)
        k4 = rhs(v + dt * k3)
        v_next = v + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if v_next >= v_threshold:
            spikes.append(1)
            v_next = v_reset
        else:
            spikes.append(0)
        v = v_next
        v_values.append(v)

    return _summarise({"v": v_values}, spikes)


def _hindmarsh_rose_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return classical-RK4 features for the driven Hindmarsh-Rose flow.

    The Hindmarsh-Rose (1984) cubic fast subsystem and slow adaptation variable are
    advanced with an independently re-derived simultaneous four-stage RK4 step. The
    maintained event is an upward ``x >= 1`` crossing and does not reset any state.
    Repeated multiplication preserves the source polynomial's evaluation order without
    importing either the hand model or the schema runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``x``, ``y``, and ``z`` state variables plus
        spike-count and first-spike-step features.
    """
    b = 3.0
    r = 0.001
    s = 4.0
    x_rest = -1.6
    threshold = 1.0
    x = -1.6
    y = -10.0
    z = 2.0
    x_values: list[float] = []
    y_values: list[float] = []
    z_values: list[float] = []
    spikes: list[int] = []

    def derivatives(x_state: float, y_state: float, z_state: float) -> tuple[float, float, float]:
        x2 = x_state * x_state
        x3 = x2 * x_state
        return (
            y_state - x3 + b * x2 - z_state + current,
            1.0 - 5.0 * x2 - y_state,
            r * (s * (x_state - x_rest) - z_state),
        )

    for _ in range(steps):
        x_prev = x
        k1 = derivatives(x, y, z)
        k2 = derivatives(
            x + 0.5 * dt * k1[0],
            y + 0.5 * dt * k1[1],
            z + 0.5 * dt * k1[2],
        )
        k3 = derivatives(
            x + 0.5 * dt * k2[0],
            y + 0.5 * dt * k2[1],
            z + 0.5 * dt * k2[2],
        )
        k4 = derivatives(x + dt * k3[0], y + dt * k3[1], z + dt * k3[2])
        x = x + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0])
        y = y + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1])
        z = z + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2])
        spikes.append(1 if (x >= threshold and x_prev < threshold) else 0)
        x_values.append(x)
        y_values.append(y)
        z_values.append(z)

    return _summarise({"x": x_values, "y": y_values, "z": z_values}, spikes)


def _morris_lecar_rk4_features(*, current: float, dt: float, steps: int) -> dict[str, float]:
    """Return exact classical-RK4 features for the driven Morris-Lecar oscillator.

    The Morris-Lecar (1981) calcium-potassium oscillator is the faithful
    conductance model: a genuine relaxation oscillator whose spikes are upward
    ``v >= v_threshold`` crossings, integrated with the same four-stage classical
    RK4 step the maintained ``MorrisLecarNeuron`` uses, with **no reset**. The
    sigmoidal calcium activation and potassium gating rate functions are transcribed
    verbatim from the schema, reusing ``numpy.tanh`` and ``numpy.cosh`` so the
    recurrence reproduces the schema runner bit-for-bit (the input current enters at
    every RK4 stage). The reference is an independent re-derivation of the committed
    driven-oscillation trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Simulation timestep.
    steps:
        Number of timesteps to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v`` and ``w`` state variables plus
        spike-count and first-spike-step features.
    """
    c_m = 20.0
    g_ca = 4.0
    g_k = 8.0
    g_l = 2.0
    e_ca = 120.0
    e_k = -84.0
    e_l = -60.0
    v1 = -1.2
    v2 = 18.0
    v3 = 12.0
    v4 = 17.4
    phi = 0.06666666666666667
    v_threshold = 0.0
    v = -60.0
    w = 0.0
    v_values: list[float] = []
    w_values: list[float] = []
    spikes: list[int] = []

    def deriv(v_state: float, w_state: float) -> tuple[float, float]:
        dv = (
            -g_ca * 0.5 * (1 + float(np.tanh((v_state - v1) / v2))) * (v_state - e_ca)
            - g_k * w_state * (v_state - e_k)
            - g_l * (v_state - e_l)
            + current
        ) / c_m
        dw = (
            phi
            * float(np.cosh((v_state - v3) / (2 * v4)))
            * (0.5 * (1 + float(np.tanh((v_state - v3) / v4))) - w_state)
        )
        return dv, dw

    for _ in range(steps):
        v_prev = v
        k1v, k1w = deriv(v, w)
        k2v, k2w = deriv(v + 0.5 * dt * k1v, w + 0.5 * dt * k1w)
        k3v, k3w = deriv(v + 0.5 * dt * k2v, w + 0.5 * dt * k2w)
        k4v, k4w = deriv(v + dt * k3v, w + dt * k3w)
        v = v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0
        w = w + dt * (k1w + 2.0 * k2w + 2.0 * k3w + k4w) / 6.0
        # Rising-edge crossing: fires when the post-step membrane is at/above threshold
        # and the previous committed membrane was below it (matching the hand model's
        # ``v >= thr and v_prev < thr`` edge test); no reset for this oscillator.
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        v_values.append(v)
        w_values.append(w)

    return _summarise({"v": v_values, "w": w_values}, spikes)


def _np_exp(x: float) -> float:
    """Return ``exp(x)`` through the same numpy implementation the schema runner uses.

    Parameters
    ----------
    x:
        Exponent argument.

    Returns
    -------
    float
        ``numpy.exp(x)`` as a Python float, bit-identical to the runner's rate terms.
    """
    return float(np.exp(x))


def _reference_exprel(x: float) -> float:
    """Return ``exprel(x) = (exp(x) - 1) / x`` with the removable-singularity limit.

    Mirrors ``EquationNeuron``'s vectorised ``exprel`` bit-for-bit: the ``|x| < 1e-9``
    branch returns the ``exprel(0) = 1`` limit as ``1 + x / 2``, and the regular
    branch uses ``numpy.expm1`` so conductance rate functions written as
    ``a / exprel(...)`` reproduce the runner exactly.

    Parameters
    ----------
    x:
        Rate-function argument.

    Returns
    -------
    float
        The exprel value matching the schema runner.
    """
    if abs(x) < 1e-9:
        return 1.0 + x / 2.0
    return float(np.expm1(x)) / x


def _hodgkin_huxley_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Hodgkin-Huxley oscillator.

    The Hodgkin-Huxley (1952) model is the faithful representation of the maintained
    ``HodgkinHuxleyNeuron(integrator="rk4")``, whose ``step()`` is itself a 100-sub-step
    macro step: each macro step advances ``substeps`` inner four-stage classical RK4
    sub-steps of ``dt`` over the same simultaneous derivative, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary, with **no reset**. The four-state membrane and Na/K
    gating rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` (the exprel-rewritten ``alpha_m`` /
    ``alpha_n``) so the recurrence reproduces the schema runner bit-for-bit. The
    reference is an independent re-derivation of the committed driven-spiking trace, not a
    copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, and ``n`` state variables
        plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 36.0
    g_l = 0.3
    e_na = 50.0
    e_k = -77.0
    e_l = -54.4
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n = sv
        dv = (
            -g_na * m**3 * h * (v - e_na) - g_k * n**4 * (v - e_k) - g_l * (v - e_l) + current
        ) / c_m
        dm = 1.0 / _reference_exprel(-(v + 40) / 10) * (1 - m) - 4 * _np_exp(-(v + 65) / 18) * m
        dh = 0.07 * _np_exp(-(v + 65) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 35) / 10)) * h
        dn = 0.1 / _reference_exprel(-(v + 55) / 10) * (1 - n) - 0.125 * _np_exp(-(v + 65) / 80) * n
        return dv, dm, dh, dn

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(4))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(4))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(4))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(4))

    state: tuple[float, ...] = (-65.0, 0.05, 0.6, 0.32)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)


def _connor_stevens_macrostep_rk4_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step RK4 features for the driven Connor-Stevens oscillator.

    The Connor-Stevens (1971) A-current model is the faithful representation of the
    maintained ``ConnorStevensNeuron`` (RK4, sub-stepped): each macro step advances
    ``substeps`` inner four-stage classical RK4 sub-steps of ``dt``, and the rising-edge
    ``v >= 0`` crossing is evaluated only on the macro boundary against the condition at
    the previous macro boundary — matching the hand model's 100-sub-step-per-millisecond
    macro step and ``EquationNeuron.step``'s macro crossing, with **no reset**. The
    six-state membrane and Na/K/A-type gating rate functions are transcribed verbatim from
    the schema, reusing :func:`_np_exp` and :func:`_reference_exprel` (and the cube-root
    ``a``-gate) so the recurrence reproduces the schema runner bit-for-bit. The reference is
    an independent re-derivation of the committed driven-spiking trace, not a copy of the
    runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner RK4 sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``v``, ``m``, ``h``, ``n``, ``a``, and ``b``
        state variables plus spike-count and first-spike-step features.
    """
    g_na = 120.0
    g_k = 20.0
    g_a = 47.7
    g_l = 0.3
    e_na = 55.0
    e_k = -72.0
    e_a = -75.0
    e_l = -17.0
    c_m = 1.0
    v_threshold = 0.0
    recorded: dict[str, list[float]] = {"v": [], "m": [], "h": [], "n": [], "a": [], "b": []}
    spikes: list[int] = []

    def deriv(sv: tuple[float, ...]) -> tuple[float, ...]:
        v, m, h, n, a, b = sv
        dv = (
            -g_na * m**3 * h * (v - e_na)
            - g_k * n**4 * (v - e_k)
            - g_a * a**3 * b * (v - e_a)
            - g_l * (v - e_l)
            + current
        ) / c_m
        dm = (
            3.8 / _reference_exprel(-(v + 29.7) / 10) * (1 - m)
            - 15.2 * _np_exp(-(v + 54.7) / 18) * m
        )
        dh = 0.266 * _np_exp(-(v + 48) / 20) * (1 - h) - 3.8 / (1 + _np_exp(-(v + 18) / 10)) * h
        dn = (
            0.2 / _reference_exprel(-(v + 45.7) / 10) * (1 - n)
            - 0.25 * _np_exp(-(v + 55.7) / 80) * n
        )
        da = (
            (0.0761 * _np_exp((v + 94.22) / 31.84) / (1 + _np_exp((v + 1.17) / 28.93)))
            ** (1.0 / 3.0)
            - a
        ) / (0.3632 + 1.158 / (1 + _np_exp((v + 55.96) / 20.12)))
        db = (1 / (1 + _np_exp((v + 53.3) / 14.54)) ** 4 - b) / (
            1.24 + 2.678 / (1 + _np_exp((v + 50) / 16.027))
        )
        return dv, dm, dh, dn, da, db

    def rk4_substep(sv: tuple[float, ...]) -> tuple[float, ...]:
        k1 = deriv(sv)
        s1 = tuple(sv[i] + 0.5 * dt * k1[i] for i in range(6))
        k2 = deriv(s1)
        s2 = tuple(sv[i] + 0.5 * dt * k2[i] for i in range(6))
        k3 = deriv(s2)
        s3 = tuple(sv[i] + dt * k3[i] for i in range(6))
        k4 = deriv(s3)
        return tuple(sv[i] + dt * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) / 6 for i in range(6))

    state: tuple[float, ...] = (-68.0, 0.01, 0.99, 0.1, 0.5, 0.1)
    for _ in range(steps):
        v_prev = state[0]
        for _ in range(substeps):
            state = rk4_substep(state)
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (state[0] >= v_threshold and v_prev < v_threshold) else 0)
        for index, name in enumerate(("v", "m", "h", "n", "a", "b")):
            recorded[name].append(state[index])

    return _summarise(recorded, spikes)


def _wang_buzsaki_macrostep_gauss_seidel_features(
    *, current: float, dt: float, steps: int, substeps: int
) -> dict[str, float]:
    """Return exact macro-step Gauss-Seidel features for the driven Wang-Buzsaki oscillator.

    The Wang-Buzsaki (1996) fast-spiking interneuron is the faithful representation of the
    maintained ``WangBuzsakiNeuron``: each macro step advances ``substeps`` inner sequential
    (Gauss-Seidel) forward-Euler sub-steps of ``dt`` — the gating variables ``h`` and ``n``
    are updated from the old voltage first, then the membrane voltage ``v`` from the
    already-updated gates (the schema declares ``method="gauss_seidel"`` with state ordered
    ``h, n, v``). Sodium activation is instantaneous: ``m_inf = alpha_m/(alpha_m+beta_m)``
    with ``alpha_m = 1/exprel(-(v+35)/10)`` (the exprel rewrite of ``0.1*(v+35)/(1-exp(...))``)
    and ``beta_m = 4*exp(-(v+60)/18)``; the potassium rate ``alpha_n`` is likewise
    ``0.1/exprel(-(v+34)/10)``. The rising-edge ``v >= v_threshold`` crossing is evaluated
    only on the macro boundary against the condition at the previous macro boundary, with
    **no reset**. The rate functions are transcribed verbatim from the schema, reusing
    :func:`_np_exp` and :func:`_reference_exprel` so the recurrence reproduces the schema
    runner bit-for-bit. The reference is an independent re-derivation of the committed
    driven-spiking trace, not a copy of the runner.

    Parameters
    ----------
    current:
        Constant input current applied at every timestep.
    dt:
        Inner sub-step timestep.
    steps:
        Number of macro steps to advance.
    substeps:
        Number of inner Gauss-Seidel sub-steps per macro step.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``h``, ``n``, and ``v`` state variables plus
        spike-count and first-spike-step features.
    """
    phi = 5.0
    g_na = 35.0
    g_k = 9.0
    g_l = 0.1
    e_na = 55.0
    e_k = -90.0
    e_l = -65.0
    capacitance = 1.0
    v_threshold = -20.0
    h = 0.8
    n = 0.1
    v = -65.0
    recorded: dict[str, list[float]] = {"h": [], "n": [], "v": []}
    spikes: list[int] = []
    for _ in range(steps):
        v_prev = v
        for _ in range(substeps):
            # ``h`` (declared first): reads the old voltage and old ``h``.
            h = (
                h
                + phi
                * (0.07 * _np_exp(-(v + 58) / 20) * (1 - h) - 1 / (1 + _np_exp(-(v + 28) / 10)) * h)
                * dt
            )
            # ``n`` (declared second): reads the old voltage and old ``n``.
            n = (
                n
                + phi
                * (
                    0.1 / _reference_exprel(-(v + 34) / 10) * (1 - n)
                    - 0.125 * _np_exp(-(v + 44) / 80) * n
                )
                * dt
            )
            # ``v`` (declared last): reads the already-updated ``h``/``n`` and old ``v``.
            inv_exprel = 1 / _reference_exprel(-(v + 35) / 10)
            m_inf = inv_exprel / (inv_exprel + 4 * _np_exp(-(v + 60) / 18))
            v = (
                v
                + (
                    -g_na * m_inf**3 * h * (v - e_na)
                    - g_k * n**4 * (v - e_k)
                    - g_l * (v - e_l)
                    + current
                )
                / capacitance
                * dt
            )
        # Macro-boundary rising-edge crossing (matching the hand model / macro runner).
        spikes.append(1 if (v >= v_threshold and v_prev < v_threshold) else 0)
        recorded["h"].append(h)
        recorded["n"].append(n)
        recorded["v"].append(v)

    return _summarise(recorded, spikes)


def _rulkov_map_features(*, current: float, steps: int) -> dict[str, float]:
    """Return exact features for the Rulkov 2002 piecewise map iteration.

    The Rulkov (2002) fast/slow model is a discrete map, so an independent
    implementation of its three-branch fast map (rational subthreshold, spike
    plateau, hard reset) and slow drift reproduces the runner exactly — a map has no
    integration error, so independent parity is exact ground truth. Upward-crossing
    detection (post-update ``x >= 0`` with pre-update ``x < 0``) matches the hand
    model and schema runner.

    Parameters
    ----------
    current:
        Constant drive applied at every iteration.
    steps:
        Number of map iterations to advance.

    Returns
    -------
    dict of str to float
        Reference feature map for the ``x`` and ``y`` state variables plus
        spike-count and first-spike-step features.
    """
    alpha = 4.0
    sigma = -1.6
    mu = 0.001
    x = -1.0
    y = -3.0
    x_values: list[float] = []
    y_values: list[float] = []
    spikes: list[int] = []
    for _ in range(steps):
        x_previous = x
        if x <= 0:
            x_next = alpha / (1.0 - x) + y + current
        elif x < alpha + y + current:
            x_next = alpha + y + current
        else:
            x_next = -1.0
        y_next = y - mu * (x + 1.0) + mu * sigma
        x, y = x_next, y_next
        spikes.append(1 if x >= 0.0 and x_previous < 0.0 else 0)
        x_values.append(x)
        y_values.append(y)

    return _summarise({"x": x_values, "y": y_values}, spikes)
