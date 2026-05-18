# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — O(1) online-learning HDL emitter

"""Emit bounded fixed-point online-learning RTL for one synapse update lane."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from sc_neurocore.learning.online_o1 import OnlineO1Config

from ._ident import sanitize_ident

ONLINE_O1_HDL_MANIFEST_SCHEMA_VERSION = "sc-neurocore.online-o1.hdl-manifest.v1"
ONLINE_O1_RESOURCE_ESTIMATE_SCHEMA_VERSION = "sc-neurocore.online-o1.resource-estimate.v1"


@dataclass(frozen=True, slots=True)
class OnlineO1ResourceEstimate:
    """Deterministic pre-synthesis resource estimate for one online-learning block."""

    target: str
    n_synapses: int
    per_synapse_state_bits: int
    total_state_bits: int
    bram18_tiles: int
    bram36_tiles: int
    lane_ff_bits: int
    estimated_luts: int
    estimated_dsps: int
    max_update_latency_cycles: int = 1
    evidence_class: str = "pre_synthesis_estimate"
    hardware_measurement_claimed: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-ready estimate payload."""

        return {
            "schema_version": ONLINE_O1_RESOURCE_ESTIMATE_SCHEMA_VERSION,
            "target": self.target,
            "evidence_class": self.evidence_class,
            "n_synapses": self.n_synapses,
            "per_synapse_state_bits": self.per_synapse_state_bits,
            "total_state_bits": self.total_state_bits,
            "bram18_tiles": self.bram18_tiles,
            "bram36_tiles": self.bram36_tiles,
            "lane_ff_bits": self.lane_ff_bits,
            "estimated_luts": self.estimated_luts,
            "estimated_dsps": self.estimated_dsps,
            "max_update_latency_cycles": self.max_update_latency_cycles,
            "hardware_measurement_claimed": self.hardware_measurement_claimed,
        }


class OnlineO1LearningEmitter:
    """Emit a synthesisable reward-modulated STDP state machine."""

    def __init__(
        self,
        *,
        module_name: str = "sc_online_o1_reward_stdp",
        config: OnlineO1Config | None = None,
    ) -> None:
        self.module_name = sanitize_ident(module_name, context="module name")
        self.config = config if config is not None else OnlineO1Config()

    def generate(self) -> str:
        """Return Verilog for one bounded online-learning synapse lane."""

        cfg = self.config
        return "\n".join(
            [
                f"module {self.module_name} #(",
                f"    parameter integer WEIGHT_BITS = {cfg.weight_bits},",
                f"    parameter integer TRACE_BITS = {cfg.trace_bits},",
                f"    parameter integer REWARD_BITS = {cfg.reward_bits},",
                f"    parameter integer LEARNING_SHIFT = {cfg.learning_shift},",
                f"    parameter integer TRACE_DECAY_SHIFT = {cfg.trace_decay_shift}",
                ") (",
                "    input wire clk,",
                "    input wire rst_n,",
                "    input wire step_en,",
                "    input wire pre_spike,",
                "    input wire post_spike,",
                "    input wire signed [REWARD_BITS-1:0] reward,",
                "    output wire [WEIGHT_BITS-1:0] current_weight,",
                "    output reg update_done",
                ");",
                "",
                f"    localparam integer PER_SYNAPSE_STATE_BITS = {cfg.per_synapse_state_bits};",
                "",
                "    reg [WEIGHT_BITS-1:0] weight;",
                "    reg [TRACE_BITS-1:0] pre_trace;",
                "    reg [TRACE_BITS-1:0] post_trace;",
                "    reg signed [TRACE_BITS-1:0] eligibility;",
                "",
                "    reg signed [TRACE_BITS:0] eligibility_delta;",
                "    reg signed [TRACE_BITS:0] decayed_eligibility;",
                "    reg signed [TRACE_BITS:0] potentiation_trace;",
                "    reg signed [WEIGHT_BITS+TRACE_BITS+REWARD_BITS-1:0] weight_update_acc;",
                "",
                "    function automatic [TRACE_BITS-1:0] decay_trace;",
                "        input [TRACE_BITS-1:0] value;",
                "        begin",
                "            if (TRACE_DECAY_SHIFT == 0)",
                "                decay_trace = value;",
                "            else",
                "                decay_trace = value - (value >> TRACE_DECAY_SHIFT);",
                "        end",
                "    endfunction",
                "",
                "    function automatic signed [TRACE_BITS-1:0] decay_signed_trace;",
                "        input signed [TRACE_BITS-1:0] value;",
                "        reg signed [TRACE_BITS-1:0] magnitude;",
                "        begin",
                "            if (TRACE_DECAY_SHIFT == 0) begin",
                "                decay_signed_trace = value;",
                "            end else if (value >= 0) begin",
                "                decay_signed_trace = value - (value >>> TRACE_DECAY_SHIFT);",
                "            end else begin",
                "                magnitude = -value;",
                "                decay_signed_trace = -(magnitude - (magnitude >>> TRACE_DECAY_SHIFT));",
                "            end",
                "        end",
                "    endfunction",
                "",
                "    function automatic signed [TRACE_BITS-1:0] sat_eligibility;",
                "        input signed [TRACE_BITS:0] value;",
                "        begin",
                "            if (value > $signed({1'b0, {TRACE_BITS-1{1'b1}}}))",
                "                sat_eligibility = $signed({1'b0, {TRACE_BITS-1{1'b1}}});",
                "            else if (value < $signed({1'b1, {TRACE_BITS-1{1'b0}}}))",
                "                sat_eligibility = $signed({1'b1, {TRACE_BITS-1{1'b0}}});",
                "            else",
                "                sat_eligibility = value[TRACE_BITS-1:0];",
                "        end",
                "    endfunction",
                "",
                "    function automatic [WEIGHT_BITS-1:0] sat_weight;",
                "        input signed [WEIGHT_BITS+TRACE_BITS+REWARD_BITS-1:0] value;",
                "        begin",
                "            if (value < 0)",
                "                sat_weight = {WEIGHT_BITS{1'b0}};",
                "            else if (value > $signed({1'b0, {WEIGHT_BITS{1'b1}}}))",
                "                sat_weight = {WEIGHT_BITS{1'b1}};",
                "            else",
                "                sat_weight = value[WEIGHT_BITS-1:0];",
                "        end",
                "    endfunction",
                "",
                "    assign current_weight = weight;",
                "",
                "    always @(*) begin",
                "        decayed_eligibility = decay_signed_trace(eligibility);",
                "        potentiation_trace = pre_spike ? {1'b0, {TRACE_BITS{1'b1}}} : $signed({1'b0, pre_trace});",
                "        eligibility_delta = post_spike ? potentiation_trace : '0;",
                "        eligibility_delta = eligibility_delta - (pre_spike ? $signed({1'b0, post_trace}) : '0);",
                "        weight_update_acc = $signed({1'b0, weight}) + (($signed(reward) * $signed(sat_eligibility(decayed_eligibility + eligibility_delta))) >>> LEARNING_SHIFT);",
                "    end",
                "",
                "    always @(posedge clk or negedge rst_n) begin",
                "        if (!rst_n) begin",
                "            weight <= {WEIGHT_BITS{1'b0}};",
                "            pre_trace <= {TRACE_BITS{1'b0}};",
                "            post_trace <= {TRACE_BITS{1'b0}};",
                "            eligibility <= {TRACE_BITS{1'b0}};",
                "            update_done <= 1'b0;",
                "        end else begin",
                "            update_done <= 1'b0;",
                "            if (step_en) begin",
                "                pre_trace <= pre_spike ? {TRACE_BITS{1'b1}} : decay_trace(pre_trace);",
                "                post_trace <= post_spike ? {TRACE_BITS{1'b1}} : decay_trace(post_trace);",
                "                eligibility <= sat_eligibility(decayed_eligibility + eligibility_delta);",
                "                weight <= sat_weight(weight_update_acc);",
                "                update_done <= 1'b1;",
                "            end",
                "        end",
                "    end",
                "endmodule",
            ]
        )

    def estimate_resources(
        self, *, n_synapses: int, target: str = "generic"
    ) -> OnlineO1ResourceEstimate:
        """Return a conservative pre-synthesis resource estimate.

        This is a deterministic planning estimate derived from the emitted state
        fields and one update lane. It deliberately does not claim synthesis,
        place-and-route, timing, power, or board measurement evidence.
        """

        if not isinstance(n_synapses, int) or isinstance(n_synapses, bool) or n_synapses <= 0:
            raise ValueError("n_synapses must be a positive integer")
        if not target:
            raise ValueError("target must be non-empty")

        cfg = self.config
        total_state_bits = n_synapses * cfg.per_synapse_state_bits
        lane_ff_bits = (
            cfg.per_synapse_state_bits
            + 3 * (cfg.trace_bits + 1)
            + cfg.weight_bits
            + cfg.trace_bits
            + cfg.reward_bits
            + 1
        )
        uses_dsp = target.lower() not in {"ice40", "gowin_littlebee", "generic_lut_only"}
        multiplier_luts = 0 if uses_dsp else math.ceil((cfg.reward_bits * cfg.trace_bits) / 4)
        estimated_luts = (
            24 + cfg.weight_bits + 2 * cfg.trace_bits + cfg.reward_bits + multiplier_luts
        )
        return OnlineO1ResourceEstimate(
            target=target,
            n_synapses=n_synapses,
            per_synapse_state_bits=cfg.per_synapse_state_bits,
            total_state_bits=total_state_bits,
            bram18_tiles=math.ceil(total_state_bits / 18_432),
            bram36_tiles=math.ceil(total_state_bits / 36_864),
            lane_ff_bits=lane_ff_bits,
            estimated_luts=estimated_luts,
            estimated_dsps=1 if uses_dsp else 0,
        )

    def manifest(self, *, verilog_path: str, n_synapses: int | None = None) -> dict[str, Any]:
        """Return deterministic metadata for the generated learning lane."""

        annotation = self.config.to_scnir_annotation(rule_id=self.module_name)
        resource_estimate = (
            self.estimate_resources(n_synapses=n_synapses, target="artix7").as_dict()
            if n_synapses is not None
            else None
        )
        return {
            "schema_version": ONLINE_O1_HDL_MANIFEST_SCHEMA_VERSION,
            "module_name": self.module_name,
            "verilog_path": verilog_path,
            "rule_family": self.config.rule_family,
            "per_synapse_state_bits": self.config.per_synapse_state_bits,
            "state_fields": annotation["state_fields"],
            "sequence_length_independent": True,
            "hidden_history_fields": [],
            "scnir_annotation": annotation,
            "resource_estimate": resource_estimate,
        }
