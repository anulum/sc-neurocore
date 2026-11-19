# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Per-neuron and per-population FPGA RTL contract tests

"""Verify neuron-template, per-population parameter keying, co-simulation, and refusal contracts."""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import compile_network_to_fpga
from sc_neurocore.nir_bridge.fpga_aer_interconnect import build_aer_interconnect
from sc_neurocore.nir_bridge.fpga_direct_interconnect import build_direct_interconnect
from sc_neurocore.nir_bridge.fpga_folded_interconnect import (
    build_folded_interconnect,
    can_fold,
)
from sc_neurocore.nir_bridge.fpga_neuron_rtl import (
    plan_population_modules,
    same_type_timesteps_match,
)
from sc_neurocore.nir_bridge.neuron_graph import ConnectionSpec, NeuronGraph, NeuronSpec
from sc_neurocore.nir_bridge.quantise_params import QuantisedGraph, quantise_graph


def _population_graph(*populations: NeuronSpec) -> NeuronGraph:
    """Build a connectionless graph for per-population RTL generation."""
    return NeuronGraph(
        populations=list(populations),
        connections=[],
        input_pop=populations[0].name,
        output_pop=populations[-1].name,
    )


_INST_RE = re.compile(r"^\s*sc_nir_([A-Za-z0-9_]+)(?:\s*#\([^)]*\))?\s+p\d+_n\d+_inst\s*\(")


def _instantiated_neuron_module_keys(top_module: str) -> set[str]:
    """Extract every instantiated ``sc_nir_<key>`` neuron module key from ``top_module``."""
    keys: set[str] = set()
    for line in top_module.splitlines():
        match = _INST_RE.match(line)
        if match is not None:
            keys.add(match.group(1))
    return keys


def _simulate_direct_raster(
    tmp_path: Path,
    *,
    top_module: str,
    neuron_modules: dict[str, str],
    module_name: str,
    currents_q: list[int],
    n_total: int,
    steps: int,
    data_width: int = 16,
) -> list[str]:
    """Run Icarus Verilog on a compiled direct/AER network and return per-step spike_bus strings."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mask = (1 << data_width) - 1
    packed = 0
    for idx, word in enumerate(currents_q):
        packed |= (int(word) & mask) << (idx * data_width)
    flat_width = max(1, len(currents_q) * data_width)
    flat_lit = f"{flat_width}'h{packed:x}"
    tb = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb;",
            "    reg clk = 1'b0;",
            "    reg rst_n = 1'b0;",
            "    reg en = 1'b1;",
            f"    reg signed [{flat_width - 1}:0] I_ext_flat = {flat_lit};",
            f"    wire [{n_total - 1}:0] spike_bus;",
            "    always #5 clk = ~clk;",
            f"    {module_name} dut (",
            "        .clk(clk), .rst_n(rst_n), .en(en),",
            "        .I_ext_flat(I_ext_flat), .spike_bus(spike_bus)",
            "    );",
            "    integer step_idx;",
            "    initial begin",
            "        @(posedge clk); #1;",
            "        rst_n = 1'b1;",
            f"        for (step_idx = 0; step_idx < {steps}; step_idx = step_idx + 1) begin",
            "            @(posedge clk); #1;",
            '            $display("RASTER %b", spike_bus);',
            "        end",
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )
    src_paths: list[str] = []
    for key, verilog in sorted(neuron_modules.items()):
        p = tmp_path / f"neuron_{key}.v"
        p.write_text(verilog, encoding="utf-8")
        src_paths.append(str(p))
    top_path = tmp_path / f"{module_name}.v"
    top_path.write_text(top_module, encoding="utf-8")
    src_paths.append(str(top_path))
    tb_path = tmp_path / "tb.v"
    tb_path.write_text(tb, encoding="utf-8")
    src_paths.append(str(tb_path))
    out_bin = tmp_path / "sim.out"
    subprocess.run(["iverilog", "-g2012", "-o", str(out_bin), *src_paths], check=True)
    out = subprocess.run(["vvp", str(out_bin)], check=True, capture_output=True, text=True).stdout
    return [line.split()[1] for line in out.splitlines() if line.startswith("RASTER ")]


def _simulate_folded_raster(
    tmp_path: Path,
    *,
    top_module: str,
    neuron_modules: dict[str, str],
    module_name: str,
    currents_q: list[int],
    n_total: int,
    steps: int,
    data_width: int = 16,
) -> list[str]:
    """Run Icarus Verilog on a compiled folded network and return committed per-tick spike_bus strings."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    mask = (1 << data_width) - 1
    packed = 0
    for idx, word in enumerate(currents_q):
        packed |= (int(word) & mask) << (idx * data_width)
    flat_width = max(1, len(currents_q) * data_width)
    flat_lit = f"{flat_width}'h{packed:x}"
    tb = "\n".join(
        [
            "`timescale 1ns / 1ps",
            "module tb_folded;",
            "    reg clk = 1'b0;",
            "    reg rst_n = 1'b0;",
            "    reg en = 1'b1;",
            f"    reg signed [{flat_width - 1}:0] I_ext_flat = {flat_lit};",
            f"    wire [{n_total - 1}:0] spike_bus;",
            "    wire tick_done;",
            "    always #5 clk = ~clk;",
            f"    {module_name} dut (",
            "        .clk(clk), .rst_n(rst_n), .en(en),",
            "        .I_ext_flat(I_ext_flat), .spike_bus(spike_bus), .tick_done(tick_done)",
            "    );",
            "    integer ticks;",
            "    initial begin",
            "        @(posedge clk); #1;",
            "        rst_n = 1'b1;",
            "        ticks = 0;",
            f"        while (ticks < {steps}) begin",
            "            @(posedge clk); #1;",
            "            if (tick_done) begin",
            '                $display("RASTER %b", spike_bus);',
            "                ticks = ticks + 1;",
            "            end",
            "        end",
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )
    src_paths: list[str] = []
    for key, verilog in sorted(neuron_modules.items()):
        if key.endswith("_pe"):
            p = tmp_path / f"pe_{key}.v"
            p.write_text(verilog, encoding="utf-8")
            src_paths.append(str(p))
    top_path = tmp_path / f"{module_name}.v"
    top_path.write_text(top_module, encoding="utf-8")
    src_paths.append(str(top_path))
    tb_path = tmp_path / "tb_folded.v"
    tb_path.write_text(tb, encoding="utf-8")
    src_paths.append(str(tb_path))
    out_bin = tmp_path / "sim_folded.out"
    subprocess.run(["iverilog", "-g2012", "-o", str(out_bin), *src_paths], check=True)
    out = subprocess.run(["vvp", str(out_bin)], check=True, capture_output=True, text=True).stdout
    return [line.split()[1] for line in out.splitlines() if line.startswith("RASTER ")]


def test_unknown_neuron_template_fails_closed() -> None:
    """Reject unknown neuron template names before emitting RTL."""
    graph = _population_graph(NeuronSpec("mystery", "not_a_neuron", 1))

    with pytest.raises(ValueError, match="No ODE template"):
        compile_network_to_fpga(graph, interconnect="direct")


def test_unknown_template_parameter_fails_closed() -> None:
    """Reject parameter names outside the canonical template contract."""
    population = NeuronSpec(
        "layer",
        "lif",
        1,
        params={"not_a_parameter": np.array([1.0], dtype=np.float64)},
    )

    with pytest.raises(ValueError, match="not supported by the 'lif' FPGA template"):
        compile_network_to_fpga(_population_graph(population), interconnect="direct")


def test_empty_parameter_array_is_rejected() -> None:
    """Reject empty parameter arrays with an explicit ValueError."""
    population = NeuronSpec(
        "layer",
        "lif",
        1,
        params={"tau": np.array([], dtype=np.float64)},
    )

    with pytest.raises(ValueError, match="Empty parameter array"):
        compile_network_to_fpga(_population_graph(population), interconnect="direct")


def test_same_type_with_different_dynamics_emits_distinct_keyed_modules() -> None:
    """Emit distinct parameter-keyed RTL modules when same-type populations differ in dt."""
    graph = _population_graph(
        NeuronSpec("fast", "lif", 1, dt=0.5),
        NeuronSpec("slow", "lif", 1, dt=1.0),
    )

    result = compile_network_to_fpga(graph, interconnect="direct")

    assert list(result.neuron_modules) == ["lif_dt_0080", "lif_dt_0100"]
    assert "module sc_nir_lif_dt_0080" in result.neuron_modules["lif_dt_0080"]
    assert "module sc_nir_lif_dt_0100" in result.neuron_modules["lif_dt_0100"]
    assert "sc_nir_lif_dt_0080 p0_n0_inst" in result.top_module
    assert "sc_nir_lif_dt_0100 p1_n0_inst" in result.top_module


def test_folded_interconnect_refuses_distinct_timesteps_within_one_type() -> None:
    """Keep fail-closed refusal on folded interconnect when same-type populations differ in dt."""
    graph = _population_graph(
        NeuronSpec("fast", "lif", 1, dt=0.5),
        NeuronSpec("slow", "lif", 1, dt=1.0),
    )
    q = Q88(data_width=16, fraction=8)
    qgraph = quantise_graph(graph, q)

    assert can_fold(qgraph, data_width=16) is False
    with pytest.raises(ValueError, match="dt"):
        compile_network_to_fpga(graph, interconnect="folded")
    with pytest.raises(ValueError, match="outside the folded interconnect"):
        build_folded_interconnect("folded_dt_mismatch", qgraph, data_width=16, fraction=8)

    delayed_ext_graph = NeuronGraph(
        populations=[NeuronSpec("target", "lif", 1, dt=1.0)],
        connections=[
            ConnectionSpec("stim", "target", np.ones((1, 1), dtype=np.float64), delay_steps=1)
        ],
        input_pop="stim",
        output_pop="target",
    )
    with pytest.raises(ValueError, match="delayed external"):
        compile_network_to_fpga(delayed_ext_graph, interconnect="folded")


def test_shared_scalar_parameter_emits_one_module_without_override() -> None:
    """Emit a single un-suffixed module when only one parameter signature exists."""
    population = NeuronSpec(
        "layer",
        "lif",
        2,
        params={"tau": np.array([20.0], dtype=np.float64)},
    )

    result = compile_network_to_fpga(_population_graph(population), interconnect="direct")

    assert list(result.neuron_modules) == ["lif"]
    assert "parameter signed [15:0] P_TAU" in result.neuron_modules["lif"]
    assert "#(" not in result.top_module


def test_explicit_default_on_later_population_needs_no_override() -> None:
    """Share one module without overrides when implicit and explicit defaults quantise identically."""
    graph = _population_graph(
        NeuronSpec("implicit", "lif", 1),
        NeuronSpec(
            "explicit",
            "lif",
            1,
            params={"tau": np.array([20.0], dtype=np.float64)},
        ),
    )

    result = compile_network_to_fpga(graph, interconnect="direct")

    assert list(result.neuron_modules) == ["lif"]
    assert ".P_TAU(" not in result.top_module


def test_quantised_equivalent_populations_share_keyed_module_without_duplicate() -> None:
    """Share one keyed module across populations whose floats quantise to identical Q8.8 words."""
    graph = _population_graph(
        NeuronSpec("p0", "lif", 1, params={"v_threshold": np.array([1.0], dtype=np.float64)}),
        NeuronSpec("p1", "lif", 1, params={"v_threshold": np.array([2.0], dtype=np.float64)}),
        NeuronSpec(
            "p2",
            "lif",
            1,
            params={"v_threshold": np.array([1.0 + 1.0e-12], dtype=np.float64)},
        ),
    )

    result = compile_network_to_fpga(graph, interconnect="direct")

    assert list(result.neuron_modules) == ["lif_v_threshold_0100", "lif_v_threshold_0200"]
    assert "sc_nir_lif_v_threshold_0100 p0_n0_inst" in result.top_module
    assert "sc_nir_lif_v_threshold_0200 p1_n0_inst" in result.top_module
    assert "sc_nir_lif_v_threshold_0100 p2_n0_inst" in result.top_module
    assert "#(" not in result.top_module


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_two_parameter_set_lif_populations_cosim_bit_exact_direct_and_folded(
    tmp_path: Path,
) -> None:
    """Verify bit-exact co-simulation for two same-type LIF populations with distinct thresholds."""
    q = Q88(data_width=16, fraction=8)
    pop_low = NeuronSpec(
        "low_thr",
        "lif",
        1,
        dt=1.0,
        params={"tau": np.array([4.0]), "v_threshold": np.array([1.0])},
    )
    pop_high = NeuronSpec(
        "high_thr",
        "lif",
        1,
        dt=1.0,
        params={"tau": np.array([4.0]), "v_threshold": np.array([2.0])},
    )
    combined = NeuronGraph(
        populations=[pop_low, pop_high],
        connections=[
            ConnectionSpec("stim", "low_thr", np.array([[1.0]], dtype=np.float64)),
            ConnectionSpec("stim", "high_thr", np.array([[1.0]], dtype=np.float64)),
        ],
        input_pop="stim",
        output_pop="high_thr",
        dt=1.0,
    )
    single_low = NeuronGraph(
        populations=[pop_low],
        connections=[ConnectionSpec("stim", "low_thr", np.array([[1.0]], dtype=np.float64))],
        input_pop="stim",
        output_pop="low_thr",
        dt=1.0,
    )
    single_high = NeuronGraph(
        populations=[pop_high],
        connections=[ConnectionSpec("stim", "high_thr", np.array([[1.0]], dtype=np.float64))],
        input_pop="stim",
        output_pop="high_thr",
        dt=1.0,
    )

    direct_res = compile_network_to_fpga(
        combined, module_name="two_set_direct", interconnect="direct"
    )
    folded_res = compile_network_to_fpga(
        combined, module_name="two_set_folded", interconnect="folded"
    )
    low_res = compile_network_to_fpga(single_low, module_name="ref_low", interconnect="direct")
    high_res = compile_network_to_fpga(single_high, module_name="ref_high", interconnect="direct")

    drive = [q.encode(3.0)]
    steps = 24
    combined_direct = _simulate_direct_raster(
        tmp_path / "comb_direct",
        top_module=direct_res.top_module,
        neuron_modules=direct_res.neuron_modules,
        module_name="two_set_direct",
        currents_q=drive,
        n_total=2,
        steps=steps,
    )
    combined_folded = _simulate_folded_raster(
        tmp_path / "comb_folded",
        top_module=folded_res.top_module,
        neuron_modules=folded_res.neuron_modules,
        module_name="two_set_folded",
        currents_q=drive,
        n_total=2,
        steps=steps,
    )
    ref_low_raster = _simulate_direct_raster(
        tmp_path / "ref_low",
        top_module=low_res.top_module,
        neuron_modules=low_res.neuron_modules,
        module_name="ref_low",
        currents_q=drive,
        n_total=1,
        steps=steps,
    )
    ref_high_raster = _simulate_direct_raster(
        tmp_path / "ref_high",
        top_module=high_res.top_module,
        neuron_modules=high_res.neuron_modules,
        module_name="ref_high",
        currents_q=drive,
        n_total=1,
        steps=steps,
    )

    expected_combined = [f"{hi}{lo}" for lo, hi in zip(ref_low_raster, ref_high_raster)]
    assert combined_direct == expected_combined
    assert combined_folded == combined_direct
    low_spikes = sum(int(b) for b in ref_low_raster)
    high_spikes = sum(int(b) for b in ref_high_raster)
    assert low_spikes > high_spikes > 0


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_folded_default_vs_explicit_parameter_cosim_matches_direct(
    tmp_path: Path,
) -> None:
    """Co-simulate folded vs direct when one population relies on template default parameters."""
    q = Q88(data_width=16, fraction=8)
    pop_default = NeuronSpec("default_thr", "lif", 1, dt=1.0)
    pop_explicit = NeuronSpec(
        "explicit_thr",
        "lif",
        1,
        dt=1.0,
        params={"v_threshold": np.array([2.0], dtype=np.float64)},
    )
    combined = NeuronGraph(
        populations=[pop_default, pop_explicit],
        connections=[
            ConnectionSpec("stim", "default_thr", np.array([[1.0]], dtype=np.float64)),
            ConnectionSpec("stim", "explicit_thr", np.array([[1.0]], dtype=np.float64)),
        ],
        input_pop="stim",
        output_pop="explicit_thr",
        dt=1.0,
    )
    single_default = NeuronGraph(
        populations=[pop_default],
        connections=[ConnectionSpec("stim", "default_thr", np.array([[1.0]], dtype=np.float64))],
        input_pop="stim",
        output_pop="default_thr",
        dt=1.0,
    )
    single_explicit = NeuronGraph(
        populations=[pop_explicit],
        connections=[ConnectionSpec("stim", "explicit_thr", np.array([[1.0]], dtype=np.float64))],
        input_pop="stim",
        output_pop="explicit_thr",
        dt=1.0,
    )

    direct_res = compile_network_to_fpga(
        combined, module_name="def_vs_exp_direct", interconnect="direct"
    )
    folded_res = compile_network_to_fpga(
        combined, module_name="def_vs_exp_folded", interconnect="folded"
    )
    def_res = compile_network_to_fpga(single_default, module_name="ref_def", interconnect="direct")
    exp_res = compile_network_to_fpga(single_explicit, module_name="ref_exp", interconnect="direct")

    assert list(direct_res.neuron_modules) == ["lif_v_threshold_0100", "lif_v_threshold_0200"]
    assert _instantiated_neuron_module_keys(direct_res.top_module) == set(direct_res.neuron_modules)
    assert folded_res.folded_metrics is not None
    assert folded_res.folded_metrics.param_rom_bits == 0

    drive = [q.encode(6.0)]
    steps = 32
    combined_direct = _simulate_direct_raster(
        tmp_path / "def_exp_direct",
        top_module=direct_res.top_module,
        neuron_modules=direct_res.neuron_modules,
        module_name="def_vs_exp_direct",
        currents_q=drive,
        n_total=2,
        steps=steps,
    )
    combined_folded = _simulate_folded_raster(
        tmp_path / "def_exp_folded",
        top_module=folded_res.top_module,
        neuron_modules=folded_res.neuron_modules,
        module_name="def_vs_exp_folded",
        currents_q=drive,
        n_total=2,
        steps=steps,
    )
    ref_def_raster = _simulate_direct_raster(
        tmp_path / "ref_def",
        top_module=def_res.top_module,
        neuron_modules=def_res.neuron_modules,
        module_name="ref_def",
        currents_q=drive,
        n_total=1,
        steps=steps,
    )
    ref_exp_raster = _simulate_direct_raster(
        tmp_path / "ref_exp",
        top_module=exp_res.top_module,
        neuron_modules=exp_res.neuron_modules,
        module_name="ref_exp",
        currents_q=drive,
        n_total=1,
        steps=steps,
    )

    expected_combined = [f"{hi}{lo}" for lo, hi in zip(ref_def_raster, ref_exp_raster)]
    assert combined_direct == expected_combined
    assert combined_folded == combined_direct
    def_spikes = sum(int(b) for b in ref_def_raster)
    exp_spikes = sum(int(b) for b in ref_exp_raster)
    assert def_spikes > exp_spikes > 0


@pytest.mark.skipif(shutil.which("iverilog") is None, reason="iverilog not installed")
def test_multi_population_distinct_parameters_compile_and_elaborate_on_aer(
    tmp_path: Path,
) -> None:
    """Verify AER interconnect instantiates per-population keyed modules and per-neuron overrides."""
    pop_a = NeuronSpec(
        "pop_a",
        "lif",
        35,
        params={"v_threshold": np.full(35, 1.0, dtype=np.float64)},
    )
    thr_b = np.full(35, 2.0, dtype=np.float64)
    thr_b[3] = 2.5
    pop_b = NeuronSpec("pop_b", "lif", 35, params={"v_threshold": thr_b})
    graph = NeuronGraph(
        populations=[pop_a, pop_b],
        connections=[
            ConnectionSpec("stim", "pop_a", np.full((35, 2), 0.5, dtype=np.float64)),
            ConnectionSpec("pop_a", "pop_b", np.full((35, 35), 0.25, dtype=np.float64)),
        ],
        input_pop="stim",
        output_pop="pop_b",
    )

    result = compile_network_to_fpga(graph, module_name="aer_keyed")

    assert result.interconnect == "aer"
    assert list(result.neuron_modules) == ["lif_v_threshold_0100", "lif_v_threshold_0200"]
    override_lines = [
        line.strip() for line in result.top_module.splitlines() if "#(" in line and "_inst" in line
    ]
    assert len(override_lines) == 1
    assert "p1_n3_inst" in override_lines[0]
    q = Q88(data_width=16, fraction=8)
    raster = _simulate_direct_raster(
        tmp_path / "aer",
        top_module=result.top_module,
        neuron_modules=result.neuron_modules,
        module_name="aer_keyed",
        currents_q=[q.encode(1.5), q.encode(1.5)],
        n_total=70,
        steps=6,
    )
    assert len(raster) == 6


def test_many_varying_fields_use_sha256_digest_and_disambiguate_collisions() -> None:
    """Use a 12-hex SHA-256 key when >3 fields vary and suffix colliding module keys."""
    pop_a = NeuronSpec(
        "a",
        "lif",
        1,
        params={
            "tau": np.array([10.0]),
            "r": np.array([1.0]),
            "v_leak": np.array([0.0]),
            "v_threshold": np.array([1.0]),
        },
    )
    pop_b = NeuronSpec(
        "b",
        "lif",
        1,
        params={
            "tau": np.array([20.0]),
            "r": np.array([2.0]),
            "v_leak": np.array([-0.5]),
            "v_threshold": np.array([2.0]),
        },
    )
    keys, reps, _defaults = plan_population_modules([pop_a, pop_b], data_width=16, fraction=8)
    assert len(keys) == 2
    assert keys[0] != keys[1]
    assert keys[0].startswith("lif_q") and len(keys[0]) == len("lif_q") + 12
    assert keys[1].startswith("lif_q") and len(keys[1]) == len("lif_q") + 12
    assert set(reps) == set(keys)

    colliding_template = {
        "equations": ["dv/dt = -v + I"],
        "threshold": "v > v_threshold",
        "reset": "v = 0.0",
        "default_params": {"v_threshold": 1.0},
    }
    pop_colliding = NeuronSpec(
        "custom_collide",
        keys[0],
        1,
        template=colliding_template,
    )
    ckeys, creps, _ = plan_population_modules(
        [pop_colliding, pop_a, pop_b], data_width=16, fraction=8
    )
    assert ckeys[0] == keys[0]
    assert ckeys[1] == f"{keys[0]}_2"
    assert len(set(ckeys)) == 3
    assert len(creps) == 3

    for ordered_pops, custom_idx, lif_a_idx in (
        ([pop_a, pop_b, pop_colliding], 2, 0),
        ([pop_a, pop_colliding, pop_b], 1, 0),
    ):
        rkeys, rreps, _ = plan_population_modules(ordered_pops, data_width=16, fraction=8)
        assert len(set(rkeys)) == len(ordered_pops)
        assert len(rreps) == len(ordered_pops)
        assert rkeys[custom_idx] == keys[0]
        assert rkeys[lif_a_idx] == f"{keys[0]}_2"
        compiled = compile_network_to_fpga(_population_graph(*ordered_pops), interconnect="direct")
        assert len(compiled.neuron_modules) == 3
        assert _instantiated_neuron_module_keys(compiled.top_module) == set(compiled.neuron_modules)


def test_emitted_top_instantiations_match_neuron_modules_keys() -> None:
    """Verify instantiated neuron module keys in top_module equal result.neuron_modules keys."""
    aer_pop_a = NeuronSpec(
        "pop_a",
        "lif",
        35,
        params={"v_threshold": np.full(35, 1.0, dtype=np.float64)},
    )
    aer_pop_b = NeuronSpec(
        "pop_b",
        "lif",
        35,
        params={"v_threshold": np.full(35, 2.0, dtype=np.float64)},
    )
    graphs_and_modes: list[tuple[NeuronGraph, str | None]] = [
        (_population_graph(NeuronSpec("only", "lif", 2)), "direct"),
        (
            _population_graph(
                NeuronSpec("fast", "lif", 1, dt=0.5),
                NeuronSpec("slow", "lif", 1, dt=1.0),
            ),
            "direct",
        ),
        (
            _population_graph(
                NeuronSpec("default_lif", "lif", 1),
                NeuronSpec("custom_lif", "lif", 1, params={"v_threshold": np.array([2.0])}),
                NeuronSpec("if_layer", "if", 2),
            ),
            "direct",
        ),
        (
            NeuronGraph(
                populations=[aer_pop_a, aer_pop_b],
                connections=[
                    ConnectionSpec("stim", "pop_a", np.full((35, 2), 0.5, dtype=np.float64)),
                    ConnectionSpec("pop_a", "pop_b", np.full((35, 35), 0.25, dtype=np.float64)),
                ],
                input_pop="stim",
                output_pop="pop_b",
            ),
            None,
        ),
    ]
    for graph, mode in graphs_and_modes:
        res = compile_network_to_fpga(graph, interconnect=mode)
        assert _instantiated_neuron_module_keys(res.top_module) == set(res.neuron_modules)

    q = Q88(data_width=16, fraction=8)
    qgraph = quantise_graph(
        NeuronGraph(
            populations=[
                NeuronSpec("def_p", "lif", 1),
                NeuronSpec("exp_p", "lif", 1, params={"v_threshold": np.array([2.0])}),
            ],
            connections=[
                ConnectionSpec("stim", "def_p", np.array([[1.0]], dtype=np.float64)),
                ConnectionSpec("stim", "exp_p", np.array([[1.0]], dtype=np.float64)),
            ],
            input_pop="stim",
            output_pop="exp_p",
        ),
        q,
    )
    direct_top = build_direct_interconnect("direct_standalone", qgraph)
    aer_top = build_aer_interconnect("aer_standalone", qgraph)
    pe_mods, folded_top = build_folded_interconnect("folded_standalone", qgraph)
    assert _instantiated_neuron_module_keys(direct_top) == {
        "lif_v_threshold_0100",
        "lif_v_threshold_0200",
    }
    assert _instantiated_neuron_module_keys(aer_top) == {
        "lif_v_threshold_0100",
        "lif_v_threshold_0200",
    }
    assert "pe_inst_lif" in folded_top and "lif_pe" in pe_mods


def test_timestep_underflow_map_mode_and_aer_zero_width_guard() -> None:
    """Reject underflowing ODE timesteps, accept discrete-map timesteps, and guard AER width."""
    underflow_pop = NeuronSpec("tiny_dt", "lif", 1, dt=1.0e-4)
    with pytest.raises(ValueError, match="underflows in Q8.8"):
        plan_population_modules([underflow_pop], data_width=16, fraction=8)
    assert same_type_timesteps_match([underflow_pop], data_width=16, fraction=8) is False

    map_template = {
        "equations": ["dv/dt = I - v_leak"],
        "threshold": "v > v_threshold",
        "reset": "v = 0.0",
        "method": "map",
        "default_params": {"v_leak": 0.0, "v_threshold": 1.0},
    }
    map_a = NeuronSpec("ma", "custom_map", 1, dt=0.25, template=map_template)
    map_b = NeuronSpec("mb", "custom_map", 1, dt=2.0, template=map_template)
    mkeys, _mreps, _ = plan_population_modules([map_a, map_b], data_width=16, fraction=8)
    assert mkeys == ["custom_map", "custom_map"]
    assert same_type_timesteps_match([map_a, map_b], data_width=16, fraction=8) is True

    connected = QuantisedGraph(
        [NeuronSpec("target", "lif", 1)],
        [ConnectionSpec("stim", "target", np.ones((1, 1), dtype=np.int64))],
        Q88(),
        "stim",
        "target",
        1.0,
        total_neurons=1,
        total_synapses=1,
    )
    with pytest.raises(ValueError, match="literal width must be positive"):
        build_aer_interconnect("zero_width_aer", connected, data_width=0)
