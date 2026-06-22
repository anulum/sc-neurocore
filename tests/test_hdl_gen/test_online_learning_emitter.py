# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Tests for O(1) online-learning HDL emitter

from __future__ import annotations

import shutil
import subprocess

import pytest

from sc_neurocore.hdl_gen.online_learning_emitter import OnlineO1LearningEmitter
from sc_neurocore.learning.online_o1 import OnlineO1Config, OnlineO1Synapse


def test_online_o1_emitter_generates_bounded_saturating_state_machine() -> None:
    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
    )
    emitter = OnlineO1LearningEmitter(module_name="o1_reward_stdp", config=config)

    verilog = emitter.generate()

    assert "module o1_reward_stdp" in verilog
    assert "localparam integer PER_SYNAPSE_STATE_BITS = 26;" in verilog
    assert "reg [WEIGHT_BITS-1:0] weight;" in verilog
    assert "reg [TRACE_BITS-1:0] pre_trace;" in verilog
    assert "reg [TRACE_BITS-1:0] post_trace;" in verilog
    assert "reg signed [TRACE_BITS-1:0] eligibility;" in verilog
    assert "function automatic [WEIGHT_BITS-1:0] sat_weight;" in verilog
    assert "function automatic signed [TRACE_BITS-1:0] sat_eligibility;" in verilog
    assert "eligibility_delta = post_spike ? potentiation_trace : '0;" in verilog
    assert (
        "potentiation_trace = pre_spike ? {1'b0, {TRACE_BITS{1'b1}}} : "
        "$signed({1'b0, pre_trace});" in verilog
    )
    assert (
        "eligibility_delta = eligibility_delta - (pre_spike ? $signed({1'b0, post_trace}) : '0);"
        in verilog
    )
    assert "weight <= sat_weight(weight_update_acc);" in verilog
    assert "assign current_weight = weight;" in verilog

    manifest = emitter.manifest(verilog_path="rtl/o1_reward_stdp.v")
    assert manifest["schema_version"] == "sc-neurocore.online-o1.hdl-manifest.v1"
    assert manifest["module_name"] == "o1_reward_stdp"
    assert manifest["verilog_path"] == "rtl/o1_reward_stdp.v"
    assert manifest["per_synapse_state_bits"] == 26
    assert manifest["sequence_length_independent"] is True
    assert manifest["hidden_history_fields"] == []


def test_online_o1_emitter_reports_presynthesis_bram_lut_estimate() -> None:
    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
    )
    emitter = OnlineO1LearningEmitter(module_name="o1_reward_stdp", config=config)

    estimate = emitter.estimate_resources(n_synapses=1024, target="artix7")
    manifest = emitter.manifest(verilog_path="rtl/o1_reward_stdp.v", n_synapses=1024)

    assert estimate.as_dict() == {
        "schema_version": "sc-neurocore.online-o1.resource-estimate.v1",
        "target": "artix7",
        "evidence_class": "pre_synthesis_estimate",
        "n_synapses": 1024,
        "per_synapse_state_bits": 26,
        "total_state_bits": 26624,
        "bram18_tiles": 2,
        "bram36_tiles": 1,
        "lane_ff_bits": 66,
        "estimated_luts": 48,
        "estimated_dsps": 1,
        "max_update_latency_cycles": 1,
        "hardware_measurement_claimed": False,
    }
    assert manifest["resource_estimate"] == estimate.as_dict()


def test_online_o1_resource_estimate_rejects_invalid_synapse_count() -> None:
    emitter = OnlineO1LearningEmitter(config=OnlineO1Config())

    with pytest.raises(ValueError, match="n_synapses"):
        emitter.estimate_resources(n_synapses=0)


def test_online_o1_resource_estimate_rejects_empty_target() -> None:
    emitter = OnlineO1LearningEmitter(config=OnlineO1Config())

    with pytest.raises(ValueError, match="target must be non-empty"):
        emitter.estimate_resources(n_synapses=128, target="")


def test_online_o1_emitter_rejects_invalid_module_name() -> None:
    with pytest.raises(ValueError, match="module name"):
        OnlineO1LearningEmitter(module_name="bad module", config=OnlineO1Config())


def test_online_o1_emitted_rtl_matches_python_reference_on_seeded_trace(tmp_path) -> None:
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        pytest.skip("iverilog and vvp are required for online O(1) RTL parity")

    config = OnlineO1Config(
        weight_bits=8,
        trace_bits=6,
        reward_bits=4,
        learning_shift=3,
        trace_decay_shift=2,
    )
    events = [
        (1, 0, 0),
        (0, 1, 7),
        (0, 0, 7),
        (0, 0, 7),
        (0, 0, -7),
        (1, 0, 0),
        (0, 1, -7),
    ]
    synapse = OnlineO1Synapse(config=config, initial_weight=0)
    expected = [
        synapse.step(pre_spike=bool(pre), post_spike=bool(post), reward=reward).weight
        for pre, post, reward in events
    ]

    module_name = "o1_reward_stdp_parity"
    rtl_path = tmp_path / "online_o1.v"
    tb_path = tmp_path / "tb.v"
    out_path = tmp_path / "tb.out"
    rtl_path.write_text(
        OnlineO1LearningEmitter(module_name=module_name, config=config).generate(),
        encoding="utf-8",
    )
    tb_path.write_text(_online_o1_testbench(module_name, events), encoding="utf-8")

    compile_result = subprocess.run(
        ["iverilog", "-g2012", "-o", str(out_path), str(rtl_path), str(tb_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stderr

    run_result = subprocess.run(
        ["vvp", str(out_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stderr

    observed = []
    for line in run_result.stdout.splitlines():
        if not line.startswith("weight "):
            continue
        _, value = line.split()
        observed.append(int(value))

    assert observed == expected


def _online_o1_testbench(module_name: str, events: list[tuple[int, int, int]]) -> str:
    event_lines = []
    for pre, post, reward in events:
        reward_literal = f"-4'sd{abs(reward)}" if reward < 0 else f"4'sd{reward}"
        event_lines.extend(
            [
                f"        pre_spike = 1'b{pre}; post_spike = 1'b{post}; reward = {reward_literal};",
                "        #1 clk = 1'b1;",
                '        #1 $display("weight %0d", current_weight);',
                "        #1 clk = 1'b0;",
            ]
        )

    return "\n".join(
        [
            "module tb;",
            "    reg clk = 1'b0;",
            "    reg rst_n = 1'b0;",
            "    reg step_en = 1'b0;",
            "    reg pre_spike = 1'b0;",
            "    reg post_spike = 1'b0;",
            "    reg signed [3:0] reward = 4'sd0;",
            "    wire [7:0] current_weight;",
            "    wire update_done;",
            "",
            f"    {module_name} uut (",
            "        .clk(clk),",
            "        .rst_n(rst_n),",
            "        .step_en(step_en),",
            "        .pre_spike(pre_spike),",
            "        .post_spike(post_spike),",
            "        .reward(reward),",
            "        .current_weight(current_weight),",
            "        .update_done(update_done)",
            "    );",
            "",
            "    initial begin",
            "        #1 clk = 1'b1;",
            "        #1 clk = 1'b0;",
            "        rst_n = 1'b1;",
            "        step_en = 1'b1;",
            *event_lines,
            "        $finish;",
            "    end",
            "endmodule",
            "",
        ]
    )
