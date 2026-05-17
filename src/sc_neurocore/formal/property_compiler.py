# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Network-level formal property compiler

from __future__ import annotations

from .network_properties import DenseLIFNetworkSpec, NetworkRateBound, NetworkRefractoryInvariant


def compile_dense_lif_fixture_rtl(network: DenseLIFNetworkSpec) -> str:
    """Compile a deterministic dense LIF fixture RTL module for formal runs."""
    max_index = max(network.input_width - 1, network.output_width - 1)
    drive_width = max(2, network.input_width.bit_length() + 1)
    threshold = 1 << max(0, min(network.state_width - 2, 8))
    leak = max(1, threshold // 8)
    drive_scale = max(1, threshold // max(1, network.input_width))

    lines = [
        "// SC-NeuroCore generated dense LIF formal fixture RTL",
        "// Deterministic fixture used by sc-neurocore formal verify-network.",
        f"module {network.name} (",
        f"    input logic {network.clock_name},",
        f"    input logic {network.reset_name},",
        f"    input logic {network.timestep_name},",
        f"    input logic [{network.input_width - 1}:0] spike_in,",
        f"    output logic [{network.output_width - 1}:0] {network.output_signal}",
        ");",
        f"    localparam logic signed [{network.state_width - 1}:0] SCNC_THRESHOLD = {network.state_width}'sd{threshold};",
        f"    localparam logic signed [{network.state_width - 1}:0] SCNC_LEAK = {network.state_width}'sd{leak};",
        f"    localparam logic signed [{network.state_width - 1}:0] SCNC_DRIVE_SCALE = {network.state_width}'sd{drive_scale};",
        "",
        f"    logic signed [{network.state_width - 1}:0] membrane [0:{network.output_width - 1}];",
        f"    logic [{drive_width - 1}:0] dense_drive [0:{network.output_width - 1}];",
        f"    logic signed [{network.state_width - 1}:0] next_membrane [0:{network.output_width - 1}];",
        "",
        "    always_comb begin",
    ]
    for output_index in range(network.output_width):
        taps = [f"spike_in[{input_index}]" for input_index in range(network.input_width)]
        lines.append(f"        dense_drive[{output_index}] = {' + '.join(taps)};")
    lines.extend(
        [
            "    end",
            "",
            f"    always_ff @(posedge {network.clock_name} or negedge {network.reset_name}) begin",
            f"        if (!{network.reset_name}) begin",
        ]
    )
    for output_index in range(network.output_width):
        lines.append(f"            membrane[{output_index}] <= '0;")
        lines.append(f"            {network.output_signal}[{output_index}] <= 1'b0;")
    lines.extend(
        [
            f"        end else if ({network.timestep_name}) begin",
        ]
    )
    for output_index in range(network.output_width):
        lines.extend(
            [
                (
                    f"            next_membrane[{output_index}] = membrane[{output_index}] "
                    f"+ ($signed({{1'b0, dense_drive[{output_index}]}}) * SCNC_DRIVE_SCALE) "
                    "- SCNC_LEAK;"
                ),
                f"            if (next_membrane[{output_index}] >= SCNC_THRESHOLD) begin",
                f"                membrane[{output_index}] <= '0;",
                f"                {network.output_signal}[{output_index}] <= 1'b1;",
                "            end else begin",
                f"                membrane[{output_index}] <= next_membrane[{output_index}];",
                f"                {network.output_signal}[{output_index}] <= 1'b0;",
                "            end",
            ]
        )
    lines.extend(
        [
            "        end else begin",
        ]
    )
    for output_index in range(network.output_width):
        lines.append(f"            {network.output_signal}[{output_index}] <= 1'b0;")
    lines.extend(
        [
            "        end",
            "    end",
            "endmodule",
            "",
        ]
    )
    return "\n".join(lines)


def compile_network_rate_bound_sva(
    network: DenseLIFNetworkSpec,
    rate_bound: NetworkRateBound,
) -> str:
    """Compile a network output spike-rate contract into deterministic SVA."""
    if rate_bound.output_index >= network.output_width:
        raise ValueError("output_index must refer to an existing network output")

    module_name = f"{network.name}_rate_bound_sva"
    assertion_name = f"a_{rate_bound.name}"
    bind_ports = (
        f".{network.clock_name}({network.clock_name}), "
        f".{network.reset_name}({network.reset_name}), "
        f".{network.timestep_name}({network.timestep_name}), "
        f".{network.output_signal}({network.output_signal})"
    )

    return "\n".join(
        [
            "// SC-NeuroCore network rate-bound SVA",
            "// Generated from a validated DenseLIFNetworkSpec and NetworkRateBound.",
            f"module {module_name} (",
            f"    input logic {network.clock_name},",
            f"    input logic {network.reset_name},",
            f"    input logic {network.timestep_name},",
            f"    input logic [{network.output_width - 1}:0] {network.output_signal}",
            ");",
            f"    parameter int unsigned SCNC_WINDOW_CYCLES = {rate_bound.window_cycles};",
            f"    parameter int unsigned SCNC_MAX_SPIKES = {rate_bound.max_spikes};",
            "",
            "    logic [$clog2(SCNC_WINDOW_CYCLES + 1)-1:0] scnc_window_count;",
            "    logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_spike_count;",
            "    logic [$clog2(SCNC_MAX_SPIKES + 2)-1:0] scnc_next_spike_count;",
            f"    wire scnc_monitored_spike = {network.output_signal}[{rate_bound.output_index}];",
            "    assign scnc_next_spike_count = scnc_spike_count + scnc_monitored_spike;",
            "",
            f"    always_ff @(posedge {network.clock_name} or negedge {network.reset_name}) begin",
            f"        if (!{network.reset_name}) begin",
            "            scnc_window_count <= '0;",
            "            scnc_spike_count <= '0;",
            f"        end else if ({network.timestep_name}) begin",
            "            if (scnc_window_count == SCNC_WINDOW_CYCLES - 1) begin",
            "                scnc_window_count <= '0;",
            "                scnc_spike_count <= scnc_monitored_spike ? 1'b1 : '0;",
            "            end else begin",
            "                scnc_window_count <= scnc_window_count + 1'b1;",
            "                if (scnc_monitored_spike && scnc_spike_count <= SCNC_MAX_SPIKES) begin",
            "                    scnc_spike_count <= scnc_spike_count + 1'b1;",
            "                end",
            "            end",
            "        end",
            "    end",
            "",
            f"    always_ff @(posedge {network.clock_name}) begin",
            f"        if ({network.reset_name} && {network.timestep_name}) begin",
            f"            {assertion_name}: assert (scnc_next_spike_count <= SCNC_MAX_SPIKES);",
            "        end",
            "    end",
            "endmodule",
            "",
            f"bind {network.name} {module_name} {module_name}_i ({bind_ports});",
            "",
        ]
    )


def compile_network_refractory_sva(
    network: DenseLIFNetworkSpec,
    refractory: NetworkRefractoryInvariant,
) -> str:
    """Compile a network output refractory contract into deterministic SVA."""
    if refractory.output_index >= network.output_width:
        raise ValueError("output_index must refer to an existing network output")

    module_name = f"{network.name}_refractory_sva"
    assertion_name = f"a_{refractory.name}"
    bind_ports = (
        f".{network.clock_name}({network.clock_name}), "
        f".{network.reset_name}({network.reset_name}), "
        f".{network.timestep_name}({network.timestep_name}), "
        f".{network.output_signal}({network.output_signal})"
    )

    return "\n".join(
        [
            "// SC-NeuroCore network refractory-invariant SVA",
            "// Generated from a validated DenseLIFNetworkSpec and NetworkRefractoryInvariant.",
            f"module {module_name} (",
            f"    input logic {network.clock_name},",
            f"    input logic {network.reset_name},",
            f"    input logic {network.timestep_name},",
            f"    input logic [{network.output_width - 1}:0] {network.output_signal}",
            ");",
            f"    parameter int unsigned SCNC_REFRACTORY_CYCLES = {refractory.refractory_cycles};",
            "",
            "    logic [$clog2(SCNC_REFRACTORY_CYCLES + 1)-1:0] scnc_refractory_count;",
            f"    wire scnc_monitored_spike = {network.output_signal}[{refractory.output_index}];",
            "    wire scnc_refractory_active = scnc_refractory_count != '0;",
            "",
            f"    always_ff @(posedge {network.clock_name} or negedge {network.reset_name}) begin",
            f"        if (!{network.reset_name}) begin",
            "            scnc_refractory_count <= '0;",
            f"        end else if ({network.timestep_name}) begin",
            "            if (scnc_monitored_spike) begin",
            "                scnc_refractory_count <= SCNC_REFRACTORY_CYCLES;",
            "            end else if (scnc_refractory_active) begin",
            "                scnc_refractory_count <= scnc_refractory_count - 1'b1;",
            "            end",
            "        end",
            "    end",
            "",
            f"    always_ff @(posedge {network.clock_name}) begin",
            f"        if ({network.reset_name} && {network.timestep_name} && scnc_refractory_active) begin",
            f"            {assertion_name}: assert (!scnc_monitored_spike);",
            "        end",
            "    end",
            "endmodule",
            "",
            f"bind {network.name} {module_name} {module_name}_i ({bind_ports});",
            "",
        ]
    )
