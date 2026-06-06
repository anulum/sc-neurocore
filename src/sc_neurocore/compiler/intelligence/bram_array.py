# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — BRAM-backed neuron array

"""Time-multiplexed BRAM-backed neuron array generation."""

from __future__ import annotations


def generate_bram_array(
    module_name: str = "sc_neuron_array",
    *,
    neuron_count: int = 1024,
    data_width: int = 16,
    state_vars: int = 1,
) -> str:
    """Generate a time-multiplexed BRAM-backed neuron array in Verilog.

    A single compute pipeline is shared across N neurons with BRAM-backed
    state. The array processes one neuron per clock cycle.

    Parameters
    ----------
    module_name : str
        Module name.
    neuron_count : int
        Number of neurons.
    data_width : int
        Fixed-point data width.
    state_vars : int
        State variables per neuron.

    Returns
    -------
    str
        Verilog module source code.
    """
    idx_w = max(1, (neuron_count - 1).bit_length())
    total_state_w = data_width * state_vars

    return f"""// Auto-generated time-multiplexed neuron array: {module_name}
// SC-NeuroCore network-level compilation
// Neurons: {neuron_count}, State width: {total_state_w}b, Pipeline: 1 neuron/cycle

module {module_name} (
    input  wire                     clk,
    input  wire                     rst,
    input  wire                     en,

    // Input current (broadcast or per-neuron)
    input  wire signed [{data_width - 1}:0]    I_global,

    // Per-neuron spike output
    output wire                     spike_out,
    output wire [{idx_w - 1}:0]              spike_neuron_id,
    output wire                     tick_done
);

    // ── BRAM state storage ──────────────────────────────────
    (* ram_style = \"block\" *)
    reg [{total_state_w - 1}:0] state_bram [0:{neuron_count - 1}];

    reg [{idx_w - 1}:0] neuron_idx;
    reg tick_active;

    reg signed [{data_width - 1}:0] v_curr;
    wire signed [{data_width - 1}:0] v_next;
    wire spike_w;

    // ── Time-multiplexed current-based LIF datapath ─────────
    // v_next = v + I/16 - v/8. Spike resets the stored membrane value to 0.
    assign v_next = v_curr + (I_global >>> 4) - (v_curr >>> 3);
    assign spike_w = (v_next > {data_width}'sd{(1 << (data_width - 2)) - 1});

    assign spike_out = spike_w & tick_active;
    assign spike_neuron_id = neuron_idx;
    assign tick_done = (neuron_idx == {idx_w}'d0) & ~tick_active;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            neuron_idx  <= 0;
            tick_active <= 1'b0;
            v_curr      <= 0;
        end else if (en) begin
            if (!tick_active) begin
                // Start new tick
                tick_active <= 1'b1;
                neuron_idx  <= 0;
                v_curr      <= state_bram[0][{data_width - 1}:0];
            end else begin
                // Write back computed state
                state_bram[neuron_idx][{data_width - 1}:0] <=
                    spike_w ? {data_width}'sd0 : v_next;

                if (neuron_idx == {idx_w}'d{neuron_count - 1}) begin
                    tick_active <= 1'b0;
                end else begin
                    neuron_idx <= neuron_idx + 1'b1;
                    v_curr <= state_bram[neuron_idx + 1'b1][{data_width - 1}:0];
                end
            end
        end
    end

endmodule
"""
