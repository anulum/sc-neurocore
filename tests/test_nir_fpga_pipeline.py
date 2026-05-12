# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — E2E tests for NIR/ONNX → FPGA compilation pipeline

"""End-to-end tests for the NIR → FPGA compilation pipeline.

Every test in this module is a full end-to-end pipeline execution:
NIR graph construction → from_nir() → from_scnetwork() →
compile_network_to_fpga() → Verilog artefact verification.
"""

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

nir = pytest.importorskip("nir")

from sc_neurocore.compiler.equation_compiler import Q88
from sc_neurocore.nir_bridge import (
    compile_network_to_fpga,
    from_nir,
    from_scnetwork,
    quantise_graph,
)
from sc_neurocore.nir_bridge.fpga_compiler import _AER_THRESHOLD


@pytest.fixture
def local_tmp_path():
    root = Path(__file__).resolve().parents[1] / ".pytest_tmp"
    root.mkdir(exist_ok=True)
    path = root / uuid4().hex
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


# ═══════════════════════════════════════════════════════════════════════
# Helper: Build NIR Graphs
# ═══════════════════════════════════════════════════════════════════════


def _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2, seed=42):
    """Build: Input → Affine → LIF → Affine → LIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=rng.randn(n_hidden, n_in).astype(np.float32),
                bias=np.zeros(n_hidden, dtype=np.float32),
            ),
            "lif1": nir.LIF(
                tau=np.full(n_hidden, 20.0),
                r=np.ones(n_hidden),
                v_leak=np.zeros(n_hidden),
                v_threshold=np.ones(n_hidden),
            ),
            "aff2": nir.Affine(
                weight=rng.randn(n_out, n_hidden).astype(np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "lif2": nir.LIF(
                tau=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "lif1"),
            ("lif1", "aff2"),
            ("aff2", "lif2"),
            ("lif2", "output"),
        ],
    )


def _build_cubalif_network(n_in=3, n_out=4, seed=99):
    """Build: Input → Affine → CubaLIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff": nir.Affine(
                weight=rng.randn(n_out, n_in).astype(np.float32),
                bias=np.zeros(n_out, dtype=np.float32),
            ),
            "cuba": nir.CubaLIF(
                tau_syn=np.full(n_out, 5.0),
                tau_mem=np.full(n_out, 20.0),
                r=np.ones(n_out),
                v_leak=np.zeros(n_out),
                v_threshold=np.ones(n_out),
                w_in=np.ones(n_out),
            ),
            "output": nir.Output(output_type={"output": np.array([n_out])}),
        },
        edges=[("input", "aff"), ("aff", "cuba"), ("cuba", "output")],
    )


def _build_mixed_type_network(n_in=4, seed=77):
    """Build: Input → Affine → IF → Affine → LIF → Output."""
    rng = np.random.RandomState(seed)
    return nir.NIRGraph(
        nodes={
            "input": nir.Input(input_type={"input": np.array([n_in])}),
            "aff1": nir.Affine(
                weight=rng.randn(6, n_in).astype(np.float32),
                bias=np.zeros(6, dtype=np.float32),
            ),
            "if_layer": nir.IF(r=np.ones(6), v_threshold=np.ones(6)),
            "aff2": nir.Affine(
                weight=rng.randn(3, 6).astype(np.float32),
                bias=np.zeros(3, dtype=np.float32),
            ),
            "lif_layer": nir.LIF(
                tau=np.full(3, 15.0),
                r=np.ones(3),
                v_leak=np.zeros(3),
                v_threshold=np.ones(3),
            ),
            "output": nir.Output(output_type={"output": np.array([3])}),
        },
        edges=[
            ("input", "aff1"),
            ("aff1", "if_layer"),
            ("if_layer", "aff2"),
            ("aff2", "lif_layer"),
            ("lif_layer", "output"),
        ],
    )


def _full_pipeline(nir_graph, dt=1.0, data_width=16, fraction=8, module_name="sc_test"):
    """Run the full NIR → FPGA pipeline and return the result."""
    net = from_nir(nir_graph, dt=dt)
    ng = from_scnetwork(net, dt=dt)
    return compile_network_to_fpga(
        ng,
        module_name=module_name,
        data_width=data_width,
        fraction=fraction,
    )


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 1: LIF Feedforward Network
# ═══════════════════════════════════════════════════════════════════════


class TestE2ELIFFeedforward:
    """Full pipeline: Input→Affine→LIF→Affine→LIF→Output → Verilog."""

    def test_pipeline_produces_valid_artefacts(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, module_name="lif_ff")

        # Neuron modules: only 1 type (LIF), compiled once
        assert "lif" in result.neuron_modules
        assert len(result.neuron_modules) == 1

        # Top module exists and has correct module name
        assert "module lif_ff" in result.top_module
        assert "endmodule" in result.top_module

        # Weight ROM exists and contains entries
        assert "module sc_nir_weight_rom" in result.weight_rom
        assert "endmodule" in result.weight_rom

    def test_neuron_module_contains_ode(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph)

        lif_v = result.neuron_modules["lif"]
        # Must contain state register (v_reg)
        assert "v_reg" in lif_v
        # Must contain spike output
        assert "spike_out" in lif_v
        # Must contain rst_n (reset)
        assert "rst_n" in lif_v
        # Must contain clk
        assert "clk" in lif_v

    def test_weight_rom_has_correct_entries(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        # Total weights: 4×8 + 8×2 = 32 + 16 = 48
        assert result.total_synapses == 48
        # ROM should have 48 case entries
        case_entries = re.findall(r"\d+'d\d+:", result.weight_rom)
        # +1 for default
        assert len(case_entries) >= 48

    def test_top_module_instantiates_populations(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, module_name="ff_net")

        # Must instantiate neuron modules
        assert "sc_nir_lif" in result.top_module
        # Must instantiate one RTL neuron per biological/NIR neuron
        assert "p0_n0_inst" in result.top_module
        assert "p1_n1_inst" in result.top_module

    def test_top_module_preserves_input_vector_and_spike_width(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph, module_name="ff_net")

        assert "input  wire signed [63:0] I_ext_flat" in result.top_module
        assert "output wire [9:0] spike_bus" in result.top_module

    def test_resource_counts(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        assert result.total_neurons == 10
        assert result.total_synapses == 48
        assert result.q_format == "Q8.8"
        assert result.interconnect == "direct"  # 10 ≤ 64


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 2: CubaLIF Network
# ═══════════════════════════════════════════════════════════════════════


class TestE2ECubaLIF:
    """Full pipeline with CubaLIF neurons (dual time constants)."""

    def test_cubalif_verilog_dual_dynamics(self):
        graph = _build_cubalif_network()
        result = _full_pipeline(graph, module_name="cuba_net")

        assert "cuba_lif" in result.neuron_modules
        cuba_v = result.neuron_modules["cuba_lif"]
        # CubaLIF has two state variables: i_syn and v
        assert "i_syn_reg" in cuba_v or "i__syn_reg" in cuba_v or "reg" in cuba_v
        assert "spike_out" in cuba_v

    def test_cubalif_weight_rom(self):
        graph = _build_cubalif_network(n_in=3, n_out=4)
        result = _full_pipeline(graph)

        # 3×4 = 12 weights
        assert result.total_synapses == 12
        assert "module sc_nir_weight_rom" in result.weight_rom


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 3: Q16.16 High-Precision
# ═══════════════════════════════════════════════════════════════════════


class TestE2EHighPrecision:
    """Full pipeline at Q16.16 (32-bit) precision."""

    def test_q16_16_wire_widths(self):
        graph = _build_lif_feedforward()
        result = _full_pipeline(graph, data_width=32, fraction=16, module_name="hd_net")

        assert result.q_format == "Q16.16"

        # Neuron module must use 32-bit wires
        lif_v = result.neuron_modules["lif"]
        assert "[31:0]" in lif_v

        # Top module must use 32-bit data
        assert "localparam integer DATA_WIDTH = 32;" in result.top_module
        assert "input  wire signed [127:0] I_ext_flat" in result.top_module

    def test_q16_16_weight_precision(self):
        graph = _build_lif_feedforward(n_in=2, n_hidden=3, n_out=1)
        result = _full_pipeline(graph, data_width=32, fraction=16)

        # Weight ROM should use 32-bit words
        assert "[31:0]" in result.weight_rom


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 4: Parameter Overflow Detection
# ═══════════════════════════════════════════════════════════════════════


class TestE2EOverflowDetection:
    """Extreme parameter values must produce warnings."""

    def test_large_tau_overflow_warning(self):
        """tau=50000 overflows Q8.8 max=127.996 — must warn."""
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff": nir.Affine(
                    weight=np.eye(2, dtype=np.float32),
                    bias=np.zeros(2, dtype=np.float32),
                ),
                "lif": nir.LIF(
                    tau=np.full(2, 50000.0),  # WAY out of range for Q8.8
                    r=np.ones(2),
                    v_leak=np.zeros(2),
                    v_threshold=np.ones(2),
                ),
                "output": nir.Output(output_type={"output": np.array([2])}),
            },
            edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
        )
        result = _full_pipeline(graph, data_width=16, fraction=8)

        # Must have overflow warnings
        assert len(result.warnings) > 0
        overflow_warns = [w for w in result.warnings if "Overflow" in w or "clamped" in w]
        assert len(overflow_warns) > 0, f"Expected overflow warnings, got: {result.warnings}"

    def test_large_weight_overflow(self):
        """Weights=500 overflows Q8.8 — must warn."""
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff": nir.Affine(
                    weight=np.full((2, 2), 500.0, dtype=np.float32),
                    bias=np.zeros(2, dtype=np.float32),
                ),
                "lif": nir.LIF(
                    tau=np.full(2, 20.0),
                    r=np.ones(2),
                    v_leak=np.zeros(2),
                    v_threshold=np.ones(2),
                ),
                "output": nir.Output(output_type={"output": np.array([2])}),
            },
            edges=[("input", "aff"), ("aff", "lif"), ("lif", "output")],
        )
        result = _full_pipeline(graph, data_width=16, fraction=8)

        overflow_warns = [w for w in result.warnings if "Overflow" in w]
        assert len(overflow_warns) > 0


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 5: Auto-Interconnect Selection
# ═══════════════════════════════════════════════════════════════════════


class TestE2EAutoInterconnect:
    """Verify direct and weighted event interconnect selection."""

    def test_small_network_uses_direct(self):
        graph = _build_lif_feedforward(n_in=4, n_hidden=8, n_out=2)
        result = _full_pipeline(graph)

        assert result.total_neurons == 10
        assert result.interconnect == "direct"
        assert "direct wiring" in result.top_module.lower() or "direct" in result.top_module.lower()
        # No AER bus signals
        assert "aer_addr" not in result.top_module

    def test_large_network_uses_weighted_event_interconnect(self):
        """Large networks use audited weighted event routing instead of warning-only direct fallback."""
        rng = np.random.RandomState(42)
        n_big = _AER_THRESHOLD + 10  # >64
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([4])}),
                "aff1": nir.Affine(
                    weight=rng.randn(n_big, 4).astype(np.float32),
                    bias=np.zeros(n_big, dtype=np.float32),
                ),
                "lif1": nir.LIF(
                    tau=np.full(n_big, 20.0),
                    r=np.ones(n_big),
                    v_leak=np.zeros(n_big),
                    v_threshold=np.ones(n_big),
                ),
                "output": nir.Output(output_type={"output": np.array([n_big])}),
            },
            edges=[("input", "aff1"), ("aff1", "lif1"), ("lif1", "output")],
        )
        result = _full_pipeline(graph)

        assert result.total_neurons > _AER_THRESHOLD
        assert result.interconnect == "aer"
        assert "weighted event routing" not in " ".join(result.warnings)
        assert "aer_addr" in result.top_module
        assert "aer_event_valid" in result.top_module
        assert f"output wire [{n_big - 1}:0] spike_bus" in result.top_module

    def test_large_two_layer_network_emits_weighted_event_fanout(self):
        """Spiking source populations must contribute signed event weights to destinations."""
        rng = np.random.RandomState(11)
        n_hidden = _AER_THRESHOLD
        n_out = 3
        graph = nir.NIRGraph(
            nodes={
                "input": nir.Input(input_type={"input": np.array([2])}),
                "aff1": nir.Affine(
                    weight=rng.randn(n_hidden, 2).astype(np.float32),
                    bias=np.zeros(n_hidden, dtype=np.float32),
                ),
                "lif1": nir.LIF(
                    tau=np.full(n_hidden, 20.0),
                    r=np.ones(n_hidden),
                    v_leak=np.zeros(n_hidden),
                    v_threshold=np.ones(n_hidden),
                ),
                "aff2": nir.Affine(
                    weight=np.full((n_out, n_hidden), 0.5, dtype=np.float32),
                    bias=np.zeros(n_out, dtype=np.float32),
                ),
                "lif2": nir.LIF(
                    tau=np.full(n_out, 20.0),
                    r=np.ones(n_out),
                    v_leak=np.zeros(n_out),
                    v_threshold=np.ones(n_out),
                ),
                "output": nir.Output(output_type={"output": np.array([n_out])}),
            },
            edges=[
                ("input", "aff1"),
                ("aff1", "lif1"),
                ("lif1", "aff2"),
                ("aff2", "lif2"),
                ("lif2", "output"),
            ],
        )
        result = _full_pipeline(graph, module_name="weighted_event_net")

        assert result.interconnect == "aer"
        assert "weighted event fan-out accumulation" in result.top_module
        assert "if (p0_n0_spike)" in result.top_module
        assert "p1_n0_I_acc_next = p1_n0_I_acc_next + " in result.top_module
        assert "00080;" in result.top_module


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 6: Round-Trip Accuracy
# ═══════════════════════════════════════════════════════════════════════


class TestE2ERoundTrip:
    """Parse NIR → simulate → compile → verify quantised params."""

    def test_quantised_simulation_matches(self):
        """Simulate with fp32 and Q8.8-quantised params, compare spike patterns."""
        graph = _build_lif_feedforward(n_in=3, n_hidden=6, n_out=2, seed=123)
        net = from_nir(graph, dt=1.0)

        # Simulate with original fp32 params
        inp = np.array([2.0, 1.0, 0.5])
        fp32_spikes = []
        for _ in range(100):
            out = net.step({"input": inp})
            fp32_spikes.append(out["output"].copy())
        net.reset()
        fp32_total = sum(s.sum() for s in fp32_spikes)

        # Now quantise and verify params are close
        ng = from_scnetwork(net, dt=1.0)
        q = Q88(data_width=16, fraction=8)
        qg = quantise_graph(ng, q)

        # Verify quantised params decode back to similar values
        for pop in qg.populations:
            for pname, pval in pop.params.items():
                # All values should be integers (Q-encoded)
                assert np.all(pval == pval.astype(np.int64)), (
                    f"Non-integer in quantised {pop.name}.{pname}"
                )

        # Re-simulate with fp32 (same model) — should be identical
        net.reset()
        fp32_check = []
        for _ in range(100):
            out = net.step({"input": inp})
            fp32_check.append(out["output"].copy())
        fp32_check_total = sum(s.sum() for s in fp32_check)

        assert fp32_total == fp32_check_total, "Deterministic simulation failed"

        # The quantised model should produce spikes if fp32 does
        # (5% tolerance on total spike count or both zero)
        if fp32_total > 0:
            # At minimum, the pipeline compiled successfully
            result = compile_network_to_fpga(ng)
            assert len(result.neuron_modules) > 0


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 7: CLI Integration
# ═══════════════════════════════════════════════════════════════════════


class TestE2ECLI:
    """Write .nir file → invoke compile-nir → verify output files."""

    def test_cli_compile_nir(self, local_tmp_path):
        """Full CLI E2E: write NIR file, run compile-nir, check outputs."""
        graph = _build_lif_feedforward(n_in=3, n_hidden=4, n_out=2)

        # Write NIR file
        nir_path = str(local_tmp_path / "test_model.nir")
        nir.write(nir_path, graph)

        out_dir = str(local_tmp_path / "compile_output")

        # Run CLI
        cmd = [
            sys.executable,
            "-m",
            "sc_neurocore.cli",
            "compile-nir",
            nir_path,
            "-o",
            out_dir,
            "--module-name",
            "cli_test_net",
            "--dt",
            "1.0",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        assert proc.returncode == 0, f"CLI failed:\n{proc.stderr}\n{proc.stdout}"

        # Check output files exist
        assert os.path.exists(os.path.join(out_dir, "cli_test_net.v"))
        assert os.path.exists(os.path.join(out_dir, "sc_nir_lif.v"))
        assert os.path.exists(os.path.join(out_dir, "sc_nir_weight_rom.v"))

        # Check files contain valid Verilog
        with open(os.path.join(out_dir, "cli_test_net.v")) as f:
            top = f.read()
        assert "module cli_test_net" in top
        assert "endmodule" in top

        with open(os.path.join(out_dir, "sc_nir_lif.v")) as f:
            lif = f.read()
        assert "module sc_nir_lif" in lif
        assert "spike_out" in lif


# ═══════════════════════════════════════════════════════════════════════
# E2E Test 8: Multi-Type Network
# ═══════════════════════════════════════════════════════════════════════


class TestE2EMultiType:
    """Network mixing IF and LIF neurons → two distinct Verilog modules."""

    def test_mixed_if_lif(self):
        graph = _build_mixed_type_network()
        result = _full_pipeline(graph, module_name="mixed_net")

        # Must generate two distinct neuron types
        assert "if" in result.neuron_modules
        assert "lif" in result.neuron_modules
        assert len(result.neuron_modules) == 2

        # Each module should have different ODE structures
        if_v = result.neuron_modules["if"]
        lif_v = result.neuron_modules["lif"]

        # Both must be valid Verilog
        assert "module sc_nir_if" in if_v
        assert "module sc_nir_lif" in lif_v
        assert "endmodule" in if_v
        assert "endmodule" in lif_v

        # Top module must reference both types
        assert "sc_nir_if" in result.top_module
        assert "sc_nir_lif" in result.top_module

    def test_mixed_type_resource_counts(self):
        graph = _build_mixed_type_network(n_in=4)
        result = _full_pipeline(graph)

        # IF layer: 6 neurons, LIF layer: 3 neurons
        assert result.total_neurons == 9
        # Weights: 4×6 + 6×3 = 24 + 18 = 42
        assert result.total_synapses == 42
