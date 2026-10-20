# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# Copyright (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore - TestFullPipeline from former test_behavioral_equivalence.py

"""Focused suite: TestFullPipeline from former test_behavioral_equivalence.py."""

from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from sc_neurocore.neurons.fixed_point_lif import FixedPointBitstreamEncoder, FixedPointLIFNeuron


class TestFullPipeline:
    """End-to-end stochastic pipeline matching HDL architecture."""

    def _run_pipeline(
        self,
        x_values: list[int],
        w_values: list[int],
        n_steps: int = 512,
        leak_k: int = 10,
        gain_k: int = 256,
        y_min: int = 0,
        y_max: int = 256,
    ) -> list[tuple[int, int, int]]:
        """
        Return spikes, consumed current and neuron state on successive clock edges.

        Encoders and neurons register together, so the neuron consumes the
        preceding encoder bits. Signed interpolation truncates towards zero.
        """
        n_inputs = len(x_values)
        assert n_inputs > 0 and len(w_values) == n_inputs

        # Create decorrelated encoders (matching HDL SEED_INIT values)
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1 + i * 7) for i in range(n_inputs)]
        weight_encs = [
            FixedPointBitstreamEncoder(seed_init=0xBEEF + i * 13) for i in range(n_inputs)
        ]

        neuron = FixedPointLIFNeuron(refractory_period=0)
        trace = []
        pre_bits = [0] * n_inputs
        w_bits = [0] * n_inputs

        for _ in range(n_steps):
            # AND synapses
            post_bits = [p & w for p, w in zip(pre_bits, w_bits)]

            # Dot-product -> current (matching sc_dotproduct_to_current.v)
            count = sum(post_bits)
            product = (y_max - y_min) * count
            quotient = abs(product) // n_inputs
            I_t = y_min + (quotient if product >= 0 else -quotient)

            spike, v = neuron.step(leak_k, gain_k, I_t, noise_in=0)
            trace.append((spike, I_t, v))
            pre_bits = [enc.step(x) for enc, x in zip(input_encs, x_values)]
            w_bits = [enc.step(w) for enc, w in zip(weight_encs, w_values)]

        return trace

    def test_pipeline_produces_spikes(self) -> None:
        """High input + high weight should produce spikes."""
        # x ~ 0.8, w ~ 0.8 -> post ~ 0.64 -> strong current
        x_vals = [52428, 52428, 52428]  # ~0.8 * 65535
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=1000)
        assert sum(row[0] for row in spikes) > 0, "Pipeline should produce spikes with high input"

    def test_pipeline_no_spikes_zero_input(self) -> None:
        """Zero input should produce no spikes."""
        x_vals = [0, 0, 0]
        w_vals = [52428, 52428, 52428]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(row[0] for row in spikes) == 0, "Zero input should produce no spikes"

    def test_pipeline_no_spikes_zero_weight(self) -> None:
        """Zero weights should produce no spikes regardless of input."""
        x_vals = [65535, 65535, 65535]
        w_vals = [0, 0, 0]
        spikes = self._run_pipeline(x_vals, w_vals, n_steps=500)
        assert sum(row[0] for row in spikes) == 0, "Zero weight should produce no spikes"

    def test_pipeline_decorrelation_matters(self) -> None:
        """
        With correlated encoders (same seed), AND-gate product is biased.

        With decorrelated encoders, result should differ.
        """
        # Run with decorrelated seeds (default pipeline)
        spikes_decorr = self._run_pipeline(
            [32768, 32768, 32768],
            [32768, 32768, 32768],
            n_steps=2000,
        )

        # Run with identical seeds (simulating the old bug)
        n_inputs = 3
        input_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        weight_encs = [FixedPointBitstreamEncoder(seed_init=0xACE1) for _ in range(n_inputs)]
        neuron = FixedPointLIFNeuron(refractory_period=0)
        spikes_corr = []
        x_val, w_val = 32768, 32768
        for _ in range(2000):
            pre = [enc.step(x_val) for enc in input_encs]
            wbits = [enc.step(w_val) for enc in weight_encs]
            post = [p & w for p, w in zip(pre, wbits)]
            count = sum(post)
            I_t = (256 * count) // n_inputs
            spike, _ = neuron.step(10, 256, I_t, 0)
            spikes_corr.append(spike)

        # Correlated encoders produce identical bitstreams when x == w,
        # so AND(x,x) = x -> higher current -> more spikes.
        # Decorrelated should have fewer spikes (p*w < p when independent).
        rate_corr = sum(spikes_corr) / len(spikes_corr)
        rate_decorr = sum(row[0] for row in spikes_decorr) / len(spikes_decorr)

        assert rate_corr != rate_decorr, (
            f"Correlated ({rate_corr:.3f}) and decorrelated ({rate_decorr:.3f}) "
            "firing rates should differ, proving decorrelation matters"
        )

    @pytest.mark.parametrize("y_min,y_max", [(-20, 285), (285, -20)])
    @pytest.mark.parametrize("directory", ["hdl", "src/sc_neurocore/hdl/primitives"])
    def test_pipeline_full_trace_matches_rtl(
        self, tmp_path: Path, y_min: int, y_max: int, directory: str
    ) -> None:
        """Compare signed nonintegral currents, spikes and states with maintained RTL."""
        x_values = [42000, 21000, 53000]
        w_values = [31000, 61000, 15000]
        expected = self._run_pipeline(x_values, w_values, 256, y_min=y_min, y_max=y_max)
        # Counts one and two distinguish signed truncation from floor division.
        assert any(current not in (y_min, y_max) for _, current, _ in expected)
        root = Path(__file__).resolve().parents[1] / directory
        packed_x = sum(word << (16 * i) for i, word in enumerate(x_values))
        packed_w = sum(word << (16 * i) for i, word in enumerate(w_values))
        bench = tmp_path / "pipeline.sv"
        bench.write_text(
            f"""`timescale 1ns / 1ps
module bench;
reg clk=0, rst_n=0;
wire spike; wire signed [15:0] current;
integer cycle, consumed;
sc_dense_layer_core #(.N_INPUTS(3), .N_NEURONS(1)) dut(
    .clk(clk),.rst_n(rst_n),.start_pulse(1'b0),.stream_len(32'd256),
    .x_input_fp(48'h{packed_x:012x}),.weight_fp(48'h{packed_w:012x}),
    .y_min_fp(16'h{y_min & 65535:04x}),.y_max_fp(16'h{y_max & 65535:04x}),
    .cfg_leak(16'd10),.cfg_gain(16'd256),.I_t(current),.spikes(spike),
    .step_valid(),.run_done(),.running());
initial begin
    #1; clk=1; #1; clk=0; #1; rst_n=1;
    for(cycle=0;cycle<256;cycle=cycle+1) begin
        #1; consumed=$signed(current); clk=1; #1;
        $display("ROW %0d %0d %0d",spike,consumed,
                 $signed(dut.v_out_bus[0]));
        clk=0; #1;
    end
    $finish;
end endmodule
"""
        )
        binary = tmp_path / "pipeline.vvp"
        primitives = [
            "sc_dense_layer_core",
            "sc_bitstream_encoder",
            "sc_bitstream_synapse",
            "sc_dotproduct_to_current",
            "sc_lif_neuron",
        ]
        subprocess.run(
            [
                "iverilog",
                "-g2012",
                "-Wall",
                "-s",
                "bench",
                "-o",
                str(binary),
                str(bench),
                *(str(root / (name + ".v")) for name in primitives),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        result = subprocess.run(
            ["vvp", str(binary)], capture_output=True, text=True, check=True, timeout=30
        )
        actual = [
            tuple(int(word) for word in line.removeprefix("ROW ").split())
            for line in result.stdout.splitlines()
            if line.startswith("ROW ")
        ]
        assert actual == expected
