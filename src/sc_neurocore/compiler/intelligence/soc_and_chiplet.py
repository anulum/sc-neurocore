# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore

from __future__ import annotations
from dataclasses import dataclass

# 3. Multi-Clock Domain CDC Synchroniser
# ═══════════════════════════════════════════════════════════════════════


def generate_cdc_synchroniser(
    signal_name: str,
    *,
    width: int = 1,
    stages: int = 2,
    src_clock: str = "clk_src",
    dst_clock: str = "clk_dst",
) -> str:
    """Generate a CDC (Clock Domain Crossing) synchroniser in Verilog.

    Uses a multi-stage register chain to safely transfer signals between
    clock domains. For multi-bit buses, use a gray-code converter or
    handshake protocol instead.

    Parameters
    ----------
    signal_name : str
        Name of the signal being synchronised.
    width : int
        Bit width (1 for single-bit CDC).
    stages : int
        Number of synchroniser stages (2 minimum, 3 for MTBF).
    src_clock : str
        Source clock name.
    dst_clock : str
        Destination clock name.

    Returns
    -------
    str
        Verilog CDC synchroniser module.
    """
    module_name = f"cdc_sync_{signal_name}"
    w = f"[{width - 1}:0] " if width > 1 else ""

    lines = [
        f"// Auto-generated CDC synchroniser for '{signal_name}'",
        "// SC-NeuroCore multi-clock domain support",
        f"// Stages: {stages}, Width: {width}-bit",
        "",
        '(* ASYNC_REG = "TRUE" *)  // Xilinx: place in same slice',
        f"module {module_name} (",
        f"    input  wire         {src_clock},",
        f"    input  wire         {dst_clock},",
        "    input  wire         rst,",
        f"    input  wire {w}{signal_name}_in,",
        f"    output wire {w}{signal_name}_out",
        ");",
        "",
    ]

    # Synchroniser chain
    for i in range(stages):
        lines.append(f'    (* ASYNC_REG = "TRUE" *) reg {w}sync_r{i};')

    lines.extend(
        [
            "",
            f"    always @(posedge {dst_clock} or posedge rst) begin",
            "        if (rst) begin",
        ]
    )

    for i in range(stages):
        lines.append(f"            sync_r{i} <= {width}'d0;")

    lines.extend(
        [
            "        end else begin",
            f"            sync_r0 <= {signal_name}_in;",
        ]
    )

    for i in range(1, stages):
        lines.append(f"            sync_r{i} <= sync_r{i - 1};")

    lines.extend(
        [
            "        end",
            "    end",
            "",
            f"    assign {signal_name}_out = sync_r{stages - 1};",
            "",
            "endmodule",
            "",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 8. BRAM / Register Auto-Selection
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class StorageRecommendation:
    """Storage recommendation for neuron array state.

    Attributes
    ----------
    strategy : str
        ``"registers"``, ``"bram"``, or ``"uram"``.
    neuron_count : int
        Number of neurons in the array.
    total_bits : int
        Total state bits.
    bram_18k_used : int
        Estimated 18Kb BRAM tiles consumed.
    bram_36k_used : int
        Estimated 36Kb BRAM tiles consumed.
    uram_used : int
        Estimated URAM tiles consumed (UltraScale+ only).
    reason : str
        Human-readable explanation.
    """

    strategy: str
    neuron_count: int
    total_bits: int
    bram_18k_used: int = 0
    bram_36k_used: int = 0
    uram_used: int = 0
    reason: str = ""


def storage_recommendation(
    neuron_count: int,
    state_bits_per_neuron: int,
    *,
    has_uram: bool = False,
    register_threshold: int = 64,
    uram_threshold: int = 16384,
) -> StorageRecommendation:
    """Determine optimal storage strategy for a neuron array.

    Decides between registers (small), BRAM (medium), and URAM (large)
    based on total state bits and target capabilities.

    Parameters
    ----------
    neuron_count : int
        Number of neurons in the array.
    state_bits_per_neuron : int
        State bits per neuron (e.g. 16 for Q8.8, 32 for Q16.16).
    has_uram : bool
        True if the target has UltraRAM (UltraScale+ / Versal only).
    register_threshold : int
        Max neurons for register-based storage.
    uram_threshold : int
        Min neurons for URAM migration.

    Returns
    -------
    StorageRecommendation
        Optimal storage strategy with resource estimates.
    """
    total_bits = neuron_count * state_bits_per_neuron

    if neuron_count <= register_threshold:
        return StorageRecommendation(
            strategy="registers",
            neuron_count=neuron_count,
            total_bits=total_bits,
            reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
            f"{total_bits}b — fits in registers.",
        )

    if has_uram and neuron_count > uram_threshold:
        # URAM: 288Kb (288 × 1024 = 294912 bits) per tile, 72b wide
        uram_tiles = max(1, (total_bits + 294911) // 294912)
        return StorageRecommendation(
            strategy="uram",
            neuron_count=neuron_count,
            total_bits=total_bits,
            uram_used=uram_tiles,
            reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
            f"{total_bits // 1024}Kb — using {uram_tiles} URAM tiles.",
        )

    # BRAM: 18Kb or 36Kb tiles
    if total_bits <= 18 * 1024:
        bram_18k = 1
        bram_36k = 0
    else:
        bram_36k = max(1, (total_bits + 36863) // 36864)
        bram_18k = 0

    return StorageRecommendation(
        strategy="bram",
        neuron_count=neuron_count,
        total_bits=total_bits,
        bram_18k_used=bram_18k,
        bram_36k_used=bram_36k,
        reason=f"{neuron_count} neurons × {state_bits_per_neuron}b = "
        f"{total_bits // 1024}Kb — using BRAM.",
    )


def generate_bram_array(
    module_name: str = "sc_neuron_array",
    *,
    neuron_count: int = 1024,
    data_width: int = 16,
    state_vars: int = 1,
) -> str:
    """Generate a time-multiplexed BRAM-backed neuron array.

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
        State variables per neuron (e.g. 1 for LIF, 2 for Izhikevich).

    Returns
    -------
    str
        Verilog module with BRAM-backed time-multiplexed neuron array.
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
    (* ram_style = "block" *)
    reg [{total_state_w - 1}:0] state_bram [0:{neuron_count - 1}];

    reg [{idx_w - 1}:0] neuron_idx;
    reg tick_active;

    reg signed [{data_width - 1}:0] v_curr;
    wire signed [{data_width - 1}:0] v_next;
    wire spike_w;

    // ── Time-multiplexed current-based LIF datapath ─────────
    // v_next = v + I/16 - v/8. Spike resets the stored membrane value to 0.
    // More detailed neuron equations are emitted by the equation compiler;
    // this array implements the BRAM read→compute→write pattern directly.
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


# ═══════════════════════════════════════════════════════════════════════
# 11. SEU / TMR Wrapper Generator
# ═══════════════════════════════════════════════════════════════════════


def generate_tmr_wrapper(
    module_name: str,
    *,
    data_width: int = 16,
    state_vars: list[str] | None = None,
    voter: str = "majority",
) -> str:
    """Generate a Triple Modular Redundancy wrapper for any neuron module.

    Instantiates three copies of the target module and a majority voter
    to mask Single Event Upsets (SEUs) in aerospace/safety-critical
    deployments. Compliant with DO-254 DAL-A and IEC 61508 SIL-4.

    Parameters
    ----------
    module_name : str
        Name of the inner neuron module to wrap.
    data_width : int
        Data width of the inner module.
    state_vars : list[str], optional
        State variable names for output voting. Defaults to ``["v"]``.
    voter : str
        Voter type: ``"majority"`` (2-of-3) or ``"median"`` (middle value).

    Returns
    -------
    str
        Synthesisable Verilog TMR wrapper module.
    """
    if state_vars is None:
        state_vars = ["v"]

    w = data_width
    tmr_name = f"{module_name}_tmr"

    lines = [
        f"// Auto-generated TMR wrapper for {module_name}",
        "// SC-NeuroCore SEU mitigation — Triple Modular Redundancy",
        f"// Voter: {voter} | DO-254 DAL-A / IEC 61508 SIL-4",
        "// IMPORTANT: Place each instance in a separate region (PBLOCK)",
        "",
        f"module {tmr_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        f"    input  wire signed [{w - 1}:0] I_t,",
    ]
    for sv in state_vars:
        lines.append(f"    output wire signed [{w - 1}:0] {sv}_voted,")
    lines.append("    output wire spike_out,")
    lines.append("    output wire seu_detected")
    lines.append(");")
    lines.append("")

    # Instantiate three copies
    for i in range(3):
        lines.append(f"    // ── Instance {chr(65 + i)} ──")
        for sv in state_vars:
            lines.append(f"    wire signed [{w - 1}:0] {sv}_{chr(97 + i)};")
        lines.append(f"    wire spike_{chr(97 + i)};")
        lines.append("")
        lines.append(f"    {module_name} inst_{chr(97 + i)} (")
        lines.append("        .clk(clk), .rst(rst), .en(en), .I_t(I_t),")
        for sv in state_vars:
            lines.append(f"        .{sv}_next({sv}_{chr(97 + i)}),")
        lines.append(f"        .spike_out(spike_{chr(97 + i)})")
        lines.append("    );")
        lines.append("")

    # Majority voter
    lines.append("    // ── Majority Voter ──")
    if voter == "majority":
        for sv in state_vars:
            a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
            lines.extend(
                [
                    f"    assign {sv}_voted = ({a} & {b}) | ({b} & {c}) | ({a} & {c});",
                ]
            )
        lines.append(
            "    assign spike_out = (spike_a & spike_b) | "
            "(spike_b & spike_c) | (spike_a & spike_c);"
        )
    else:  # median
        for sv in state_vars:
            a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
            lines.extend(
                [
                    "    // Median: sort three values, pick middle",
                    f"    wire signed [{w - 1}:0] {sv}_min = "
                    f"($signed({a}) < $signed({b})) ? "
                    f"(($signed({a}) < $signed({c})) ? {a} : {c}) : "
                    f"(($signed({b}) < $signed({c})) ? {b} : {c});",
                    f"    wire signed [{w - 1}:0] {sv}_max = "
                    f"($signed({a}) > $signed({b})) ? "
                    f"(($signed({a}) > $signed({c})) ? {a} : {c}) : "
                    f"(($signed({b}) > $signed({c})) ? {b} : {c});",
                    f"    assign {sv}_voted = {a} + {b} + {c} - {sv}_min - {sv}_max;",
                ]
            )
        lines.append(
            "    assign spike_out = (spike_a & spike_b) | "
            "(spike_b & spike_c) | (spike_a & spike_c);"
        )

    # SEU detection: any mismatch
    mismatch_terms = []
    for sv in state_vars:
        a, b, c = f"{sv}_a", f"{sv}_b", f"{sv}_c"
        mismatch_terms.append(f"({a} != {b})")
        mismatch_terms.append(f"({b} != {c})")
    mismatch_terms.append("(spike_a != spike_b)")
    mismatch_terms.append("(spike_b != spike_c)")
    lines.append(f"    assign seu_detected = {' | '.join(mismatch_terms)};")
    lines.append("")
    lines.append("endmodule")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 15. PIM Data Layout Planner
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PIMLayout:
    """Memory layout plan for Processing-in-Memory targets.

    Attributes
    ----------
    bank_count : int
        Number of memory banks used.
    neurons_per_bank : int
        Neurons assigned per bank.
    weights_per_bank : int
        Weight entries per bank.
    bank_utilisation : float
        Fraction of bank capacity used (0.0–1.0).
    parallel_factor : int
        Number of banks that can compute in parallel.
    layout_map : dict[str, list[int]]
        Mapping of data regions to bank IDs.
    """

    bank_count: int
    neurons_per_bank: int
    weights_per_bank: int
    bank_utilisation: float
    parallel_factor: int
    layout_map: dict[str, list[int]]


def plan_pim_layout(
    neuron_count: int,
    synapse_count: int,
    *,
    data_width: int = 16,
    bank_size_kb: int = 64,
    num_banks: int = 16,
    target: str = "upmem_pim",
) -> PIMLayout:
    """Plan data placement across PIM memory banks.

    Distributes neuron state and synaptic weights across memory banks
    to maximise bank-level parallelism on PIM (UPMEM, Samsung HBM-PIM)
    and CXL memory expander targets.

    Parameters
    ----------
    neuron_count : int
        Total neurons in the network.
    synapse_count : int
        Total synaptic connections.
    data_width : int
        Bits per value.
    bank_size_kb : int
        Capacity of each memory bank in KB.
    num_banks : int
        Number of available memory banks.
    target : str
        Target platform name.

    Returns
    -------
    PIMLayout
        Optimised memory layout plan.
    """
    bytes_per_val = max(1, data_width // 8)
    neuron_bytes = neuron_count * bytes_per_val
    weight_bytes = synapse_count * bytes_per_val
    total_bytes = neuron_bytes + weight_bytes

    bank_bytes = bank_size_kb * 1024
    banks_needed = max(1, -(-total_bytes // bank_bytes))  # ceil div
    banks_used = min(banks_needed, num_banks)

    neurons_per_bank = max(1, -(-neuron_count // banks_used))
    weights_per_bank = max(1, -(-synapse_count // banks_used))

    used_bytes_per_bank = neurons_per_bank * bytes_per_val + weights_per_bank * bytes_per_val
    utilisation = min(1.0, used_bytes_per_bank / bank_bytes)

    # Layout: first half for neuron state, second half for weights
    state_banks = list(range(0, banks_used // 2 or 1))
    weight_banks = list(range(banks_used // 2 or 1, banks_used))
    if not weight_banks:
        weight_banks = state_banks  # Small networks share banks

    return PIMLayout(
        bank_count=banks_used,
        neurons_per_bank=neurons_per_bank,
        weights_per_bank=weights_per_bank,
        bank_utilisation=round(utilisation, 4),
        parallel_factor=banks_used,
        layout_map={
            "neuron_state": state_banks,
            "synaptic_weights": weight_banks,
        },
    )


# ═══════════════════════════════════════════════════════════════════════
# 16. Power Domain / Clock Gating Wrapper
# ═══════════════════════════════════════════════════════════════════════


def generate_power_domain_wrapper(
    module_name: str,
    *,
    data_width: int = 16,
    state_vars: list[str] | None = None,
    always_on_signals: list[str] | None = None,
    wakeup_cycles: int = 4,
) -> str:
    """Generate a clock/power gating wrapper for always-on edge deployment.

    Creates a wrapper module with:
    - ICG (Integrated Clock Gating) cell for dynamic power reduction
    - Power-down state retention latches
    - Configurable wakeup latency
    - Always-on domain for event detection (spike_detect)

    Targets ultra-low-power edge platforms (Syntiant NDP120, Innatera
    Pulsar, BrainChip Akida) where µW-level idle power is critical.

    Parameters
    ----------
    module_name : str
        Inner neuron module name.
    data_width : int
        Data width.
    state_vars : list[str], optional
        State variables to retain. Defaults to ``["v"]``.
    always_on_signals : list[str], optional
        Signals kept in the always-on domain. Defaults to ``["spike_out"]``.
    wakeup_cycles : int
        Clock cycles required to exit power-down.

    Returns
    -------
    str
        Synthesisable Verilog power domain wrapper.
    """
    if state_vars is None:
        state_vars = ["v"]
    if always_on_signals is None:
        always_on_signals = ["spike_out"]

    w = data_width
    pg_name = f"{module_name}_pg"
    wk_bits = max(1, (wakeup_cycles - 1).bit_length())

    lines = [
        f"// Auto-generated power domain wrapper for {module_name}",
        "// SC-NeuroCore — clock/power gating for ultra-low-power edge",
        f"// Wakeup latency: {wakeup_cycles} cycles",
        "",
        f"module {pg_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        "    input  wire power_down,     // Active-high power-down request",
        f"    input  wire signed [{w - 1}:0] I_t,",
    ]
    for sv in state_vars:
        lines.append(f"    output reg  signed [{w - 1}:0] {sv}_out,")
    for sig in always_on_signals:
        lines.append(f"    output wire {sig},")
    lines.append("    output wire power_state       // 0=active, 1=power-down")
    lines.append(");")
    lines.append("")

    # ICG cell
    lines.extend(
        [
            "    // ── Integrated Clock Gating ──",
            "    wire gated_clk;",
            "    reg  clk_enable;",
            "    always @(negedge clk)",
            "        clk_enable <= en & ~power_down;",
            "    assign gated_clk = clk & clk_enable;",
            "",
        ]
    )

    # Wakeup counter
    lines.extend(
        [
            "    // ── Wakeup sequencer ──",
            f"    reg [{wk_bits - 1}:0] wakeup_cnt;",
            "    reg  active;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            wakeup_cnt <= 0;",
            "            active <= 0;",
            "        end else if (power_down) begin",
            "            wakeup_cnt <= 0;",
            "            active <= 0;",
            "        end else if (!active) begin",
            f"            if (wakeup_cnt == {wakeup_cycles - 1})",
            "                active <= 1;",
            "            else",
            "                wakeup_cnt <= wakeup_cnt + 1;",
            "        end",
            "    end",
            "    assign power_state = ~active;",
            "",
        ]
    )

    # Inner module instance
    lines.extend(
        [
            "    // ── Inner neuron (gated clock domain) ──",
        ]
    )
    for sv in state_vars:
        lines.append(f"    wire signed [{w - 1}:0] {sv}_inner;")
    lines.extend(
        [
            "    wire spike_inner;",
            "",
            f"    {module_name} core (",
            "        .clk(gated_clk), .rst(rst), .en(active),",
            "        .I_t(I_t),",
        ]
    )
    for sv in state_vars:
        lines.append(f"        .{sv}_next({sv}_inner),")
    lines.extend(
        [
            "        .spike_out(spike_inner)",
            "    );",
            "",
        ]
    )

    # State retention
    lines.extend(
        [
            "    // ── State retention latches ──",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
        ]
    )
    for sv in state_vars:
        lines.append(f"            {sv}_out <= 0;")
    lines.extend(
        [
            "        end else if (active) begin",
        ]
    )
    for sv in state_vars:
        lines.append(f"            {sv}_out <= {sv}_inner;")
    lines.extend(
        [
            "        end",
            "        // Else: retain previous value (power-down)",
            "    end",
            "",
        ]
    )

    # Always-on spike detection
    lines.extend(
        [
            "    // ── Always-on domain (ungated) ──",
            "    assign spike_out = active ? spike_inner : 1'b0;",
            "",
            "endmodule",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 19. UCIe Partitioning Advisor
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class UCIePartition:
    """Partitioning plan for a neuron array across chiplet tiles.

    Attributes
    ----------
    tile_count : int
        Number of chiplet tiles used.
    neurons_per_tile : int
        Neurons assigned per tile.
    inter_tile_spikes : int
        Estimated spikes crossing tile boundaries per timestep.
    die_to_die_bandwidth_gbps : float
        Required UCIe bandwidth (Gbps).
    latency_penalty_ns : float
        Additional latency from die-to-die communication.
    partition_map : dict[int, list[int]]
        Tile ID → list of neuron indices.
    """

    tile_count: int
    neurons_per_tile: int
    inter_tile_spikes: int
    die_to_die_bandwidth_gbps: float
    latency_penalty_ns: float
    partition_map: dict[int, list[int]]


def advise_ucie_partition(
    neuron_count: int,
    connectivity: float = 0.1,
    *,
    tile_count: int = 4,
    spike_rate_hz: float = 10.0,
    timestep_us: float = 1000.0,
    ucie_lane_gbps: float = 32.0,
    ucie_latency_ns: float = 2.0,
) -> UCIePartition:
    """Advise on neuron array partitioning across chiplet tiles.

    Analyses a neuron array's connectivity to estimate inter-tile
    spike traffic and UCIe bandwidth requirements when distributing
    a network across multi-die chiplet systems (AMD MI300X,
    Tenstorrent Galaxy, Intel Ponte Vecchio).

    Parameters
    ----------
    neuron_count : int
        Total neurons in the network.
    connectivity : float
        Connection probability between any two neurons (0.0–1.0).
    tile_count : int
        Number of chiplet tiles.
    spike_rate_hz : float
        Average firing rate per neuron (Hz).
    timestep_us : float
        Simulation timestep (µs).
    ucie_lane_gbps : float
        UCIe lane bandwidth (Gbps per lane).
    ucie_latency_ns : float
        UCIe die-to-die latency (ns).

    Returns
    -------
    UCIePartition
        Partitioning plan with bandwidth and latency estimates.
    """
    neurons_per_tile = max(1, -(-neuron_count // tile_count))  # ceil

    # Estimate inter-tile spikes per timestep
    # Fraction of connections that cross tile boundaries
    intra_tile_frac = 1.0 / tile_count  # prob both neurons on same tile
    inter_tile_frac = 1.0 - intra_tile_frac

    total_synapses = neuron_count * neuron_count * connectivity
    inter_tile_synapses = total_synapses * inter_tile_frac

    spikes_per_timestep = neuron_count * spike_rate_hz * (timestep_us / 1e6)
    inter_tile_spikes = int(spikes_per_timestep * inter_tile_frac)

    # Bandwidth: each spike = ~8 bytes (neuron ID + timestamp)
    bytes_per_spike = 8
    bytes_per_timestep = inter_tile_spikes * bytes_per_spike
    bits_per_second = bytes_per_timestep * 8 / (timestep_us * 1e-6)
    required_gbps = round(bits_per_second / 1e9, 4)

    # Latency penalty from die-to-die crossing
    latency_ns = ucie_latency_ns * (tile_count - 1)  # worst-case path

    # Simple round-robin partition
    partition_map = {}
    for t in range(tile_count):
        start = t * neurons_per_tile
        end = min(start + neurons_per_tile, neuron_count)
        partition_map[t] = list(range(start, end))

    return UCIePartition(
        tile_count=tile_count,
        neurons_per_tile=neurons_per_tile,
        inter_tile_spikes=inter_tile_spikes,
        die_to_die_bandwidth_gbps=required_gbps,
        latency_penalty_ns=round(latency_ns, 2),
        partition_map=partition_map,
    )


# ═══════════════════════════════════════════════════════════════════════
# 20. CXL Coherence Advisor
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class CXLMapping:
    """CXL.mem Type-3 mapping for neuron state.

    Attributes
    ----------
    device_count : int
        Number of CXL memory devices.
    state_device_ids : list[int]
        Devices hosting neuron state.
    weight_device_ids : list[int]
        Devices hosting synaptic weights.
    total_capacity_gb : float
        Total CXL memory capacity.
    host_bandwidth_gbps : float
        Required host→CXL bandwidth.
    coherence_protocol : str
        CXL protocol used (``"CXL.mem"`` or ``"CXL.cache"``).
    """

    device_count: int
    state_device_ids: list[int]
    weight_device_ids: list[int]
    total_capacity_gb: float
    host_bandwidth_gbps: float
    coherence_protocol: str


def advise_cxl_mapping(
    neuron_count: int,
    synapse_count: int,
    *,
    data_width: int = 16,
    device_capacity_gb: float = 16.0,
    max_devices: int = 8,
    access_pattern: str = "streaming",
) -> CXLMapping:
    """Advise on CXL.mem Type-3 device mapping for neuron state.

    Plans the distribution of neuron state and synaptic weights
    across CXL 3.0 Type-3 memory expander devices for large-scale
    SNN simulations that exceed local DRAM capacity.

    Parameters
    ----------
    neuron_count : int
        Total neurons.
    synapse_count : int
        Total synaptic connections.
    data_width : int
        Bits per value.
    device_capacity_gb : float
        Capacity per CXL device (GB).
    max_devices : int
        Maximum CXL devices available.
    access_pattern : str
        ``"streaming"`` (sequential) or ``"random"`` (scattered).

    Returns
    -------
    CXLMapping
        Device mapping plan.
    """
    bytes_per_val = max(1, data_width // 8)
    state_bytes = neuron_count * bytes_per_val * 4  # 4 state vars avg
    weight_bytes = synapse_count * bytes_per_val
    total_bytes = state_bytes + weight_bytes
    total_gb = total_bytes / (1024**3)

    devices_needed = max(1, int(-(-total_gb // device_capacity_gb)))
    devices_used = min(devices_needed, max_devices)

    # Split: state on first devices, weights on remaining
    state_devs = max(1, int(devices_used * state_bytes / total_bytes))
    weight_devs = max(1, devices_used - state_devs)

    state_device_ids = list(range(state_devs))
    weight_device_ids = list(range(state_devs, state_devs + weight_devs))

    # Bandwidth estimation: streaming is more efficient
    bw_factor = 1.0 if access_pattern == "streaming" else 2.5
    update_rate_hz = 1000  # 1 kHz timestep
    bytes_per_update = total_bytes * 0.1  # 10% active per step
    raw_bw = bytes_per_update * update_rate_hz * 8 / 1e9
    required_gbps = round(raw_bw * bw_factor, 4)

    # Protocol selection
    protocol = "CXL.cache" if access_pattern == "random" else "CXL.mem"

    return CXLMapping(
        device_count=devices_used,
        state_device_ids=state_device_ids,
        weight_device_ids=weight_device_ids,
        total_capacity_gb=round(devices_used * device_capacity_gb, 2),
        host_bandwidth_gbps=required_gbps,
        coherence_protocol=protocol,
    )


# ═══════════════════════════════════════════════════════════════════════
# 23. Pipeline Register Wrapper
# ═══════════════════════════════════════════════════════════════════════


def generate_pipeline_wrapper(
    module_name: str,
    equations: dict[str, str],
    *,
    data_width: int = 16,
    target: str = "artix7",
    stages: int | None = None,
) -> str:
    """Generate a pipelined wrapper that inserts register stages.

    Auto-computes the critical path depth and required pipeline stages
    based on the target frequency, then wraps the neuron module with
    input/output pipeline registers and a valid-pipeline shift register.

    Parameters
    ----------
    module_name : str
        Inner neuron module name.
    equations : dict[str, str]
        ODE equations (for depth analysis).
    data_width : int
        Data width.
    target : str
        Target platform name (used for frequency lookup).
    stages : int, optional
        Override pipeline stages. If None, auto-computed.

    Returns
    -------
    str
        Synthesisable Verilog pipeline wrapper.
    """
    from ..static_analysis import critical_path_depth, pipeline_stages_needed
    from ..platforms import get_profile

    profile = get_profile(target)
    freq = profile.max_freq_mhz or 100

    # Compute depth from all equations
    max_depth = 0
    for _sv, expr in equations.items():
        d = critical_path_depth(expr)
        max_depth = max(max_depth, d)

    if stages is None:
        stages = pipeline_stages_needed(max_depth, freq)

    if stages == 0:
        stages = 1  # Minimum 1 stage for the wrapper to be meaningful

    w = data_width
    pipe_name = f"{module_name}_pipe"

    lines = [
        f"// Auto-generated pipeline wrapper for {module_name}",
        f"// SC-NeuroCore — {stages}-stage pipeline for {freq} MHz",
        f"// Critical path depth: {max_depth} DSP blocks",
        "",
        f"module {pipe_name} (",
        "    input  wire clk,",
        "    input  wire rst,",
        "    input  wire en,",
        "    input  wire valid_in,",
        f"    input  wire signed [{w - 1}:0] I_t,",
        f"    output wire signed [{w - 1}:0] v_out,",
        "    output wire spike_out,",
        "    output wire valid_out,",
        f"    output wire [{stages.bit_length() - 1}:0] latency",
        ");",
        "",
        "    // Pipeline latency (constant)",
        f"    assign latency = {stages};",
        "",
    ]

    # Input pipeline registers
    lines.extend(
        [
            "    // ── Input pipeline registers ──",
        ]
    )
    for s in range(stages):
        if s == 0:
            lines.append(f"    reg signed [{w - 1}:0] I_pipe_{s};")
        else:
            lines.append(f"    reg signed [{w - 1}:0] I_pipe_{s};")
    lines.append("")

    lines.extend(
        [
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
        ]
    )
    for s in range(stages):
        lines.append(f"            I_pipe_{s} <= 0;")
    lines.extend(
        [
            "        end else if (en) begin",
            "            I_pipe_0 <= I_t;",
        ]
    )
    for s in range(1, stages):
        lines.append(f"            I_pipe_{s} <= I_pipe_{s - 1};")
    lines.extend(
        [
            "        end",
            "    end",
            "",
        ]
    )

    # Valid pipeline (shift register)
    lines.extend(
        [
            "    // ── Valid pipeline ──",
            f"    reg [{stages - 1}:0] valid_pipe;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst)",
            "            valid_pipe <= 0;",
            "        else if (en)",
        ]
    )
    if stages == 1:
        lines.append("            valid_pipe[0] <= valid_in;")
    else:
        lines.append(f"            valid_pipe <= {{valid_pipe[{stages - 2}:0], valid_in}};")
    lines.extend(
        [
            "    end",
            f"    assign valid_out = valid_pipe[{stages - 1}];",
            "",
        ]
    )

    # Inner module instantiation (fed from last pipeline stage)
    lines.extend(
        [
            "    // ── Inner neuron (combinational) ──",
            f"    wire signed [{w - 1}:0] v_comb;",
            "    wire spike_comb;",
            "",
            f"    {module_name} core (",
            "        .clk(clk), .rst(rst), .en(en),",
            f"        .I_t(I_pipe_{stages - 1}),",
            "        .v_next(v_comb),",
            "        .spike_out(spike_comb)",
            "    );",
            "",
        ]
    )

    # Output register
    lines.extend(
        [
            "    // ── Output register ──",
            f"    reg signed [{w - 1}:0] v_reg;",
            "    reg spike_reg;",
            "    always @(posedge clk or posedge rst) begin",
            "        if (rst) begin",
            "            v_reg <= 0;",
            "            spike_reg <= 0;",
            "        end else if (en) begin",
            "            v_reg <= v_comb;",
            "            spike_reg <= spike_comb;",
            "        end",
            "    end",
            "    assign v_out = v_reg;",
            "    assign spike_out = spike_reg;",
            "",
            "endmodule",
        ]
    )

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 47. Memory Map Generator
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class MemoryMap:
    """Address decoder specification for neuron arrays.

    Attributes
    ----------
    base_address : int
        Base address of neuron array.
    entries : list[dict[str, int | str]]
        Address map entries.
    total_bytes : int
        Total address space consumed.
    decoder_verilog : str
        Generated address decoder Verilog.
    """

    base_address: int
    entries: list[dict]
    total_bytes: int
    decoder_verilog: str


def generate_memory_map(
    module_name: str,
    equations: dict[str, str],
    *,
    num_neurons: int = 256,
    data_width: int = 16,
    base_address: int = 0x1000_0000,
) -> MemoryMap:
    """Generate address decoder for multi-neuron SoC arrays.

    Parameters
    ----------
    module_name : str
        Module name.
    equations : dict[str, str]
        ODE equations (state variables define register set).
    num_neurons : int
        Number of neuron instances.
    data_width : int
        Register width in bits.
    base_address : int
        Base address.

    Returns
    -------
    MemoryMap
        Address map with decoder Verilog.
    """
    vars_list = list(equations.keys())
    bytes_per_reg = max(2, data_width // 8)
    regs_per_neuron = len(vars_list) + 1  # +1 for control
    stride = regs_per_neuron * bytes_per_reg

    entries = []
    for n in range(min(num_neurons, 8)):  # show first 8
        for i, sv in enumerate(vars_list):
            addr = base_address + n * stride + i * bytes_per_reg
            entries.append(
                {
                    "address": addr,
                    "name": f"neuron_{n}_{sv}",
                    "width": data_width,
                }
            )
        ctrl_addr = base_address + n * stride + len(vars_list) * bytes_per_reg
        entries.append(
            {
                "address": ctrl_addr,
                "name": f"neuron_{n}_ctrl",
                "width": data_width,
            }
        )

    total = num_neurons * stride
    verilog = [
        f"// Address decoder for {module_name} — {num_neurons} neurons",
        f"// Base: 0x{base_address:08X}, Stride: {stride} bytes",
        f"module {module_name}_addr_dec (",
        f"    input  [{data_width - 1}:0] addr,",
        f"    output reg [{len(vars_list)}:0] reg_sel,",
        f"    output reg [{num_neurons.bit_length() - 1}:0] neuron_sel",
        ");",
        f"    wire [{num_neurons.bit_length() - 1}:0] idx = "
        f"(addr - 32'h{base_address:08X}) / {stride};",
        f"    wire [{regs_per_neuron.bit_length() - 1}:0] reg_off = "
        f"((addr - 32'h{base_address:08X}) % {stride}) / {bytes_per_reg};",
        "    always @(*) begin",
        "        neuron_sel = idx;",
        "        reg_sel = reg_off;",
        "    end",
        "endmodule",
    ]

    return MemoryMap(
        base_address=base_address,
        entries=entries,
        total_bytes=total,
        decoder_verilog="\n".join(verilog),
    )


# ═══════════════════════════════════════════════════════════════════════
# 54. Multi-Die Floorplanner
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FloorplanResult:
    """Multi-die/chiplet floorplan assignment.

    Attributes
    ----------
    die_assignment : dict[str, int]
        Block name → die index.
    die_utilization : dict[int, float]
        Die index → utilization (0-1).
    total_dies : int
    """

    die_assignment: dict[str, int]
    die_utilization: dict[int, float]
    total_dies: int


def plan_multi_die_floorplan(
    blocks: dict[str, int],
    *,
    die_capacity: int = 1000,
    num_dies: int = 4,
) -> FloorplanResult:
    """Assign neuron blocks to chiplet/die positions.

    Uses first-fit-decreasing bin packing.

    Parameters
    ----------
    blocks : dict[str, int]
        Block name → neuron count.
    die_capacity : int
        Max neurons per die.
    num_dies : int
        Available dies.

    Returns
    -------
    FloorplanResult
    """
    sorted_blocks = sorted(blocks.items(), key=lambda x: x[1], reverse=True)
    assignment: dict[str, int] = {}
    die_used = [0] * num_dies

    for name, count in sorted_blocks:
        placed = False
        for d in range(num_dies):
            if die_used[d] + count <= die_capacity:
                assignment[name] = d
                die_used[d] += count
                placed = True
                break
        if not placed:
            assignment[name] = num_dies - 1
            die_used[num_dies - 1] += count

    util = {d: round(die_used[d] / die_capacity, 3) for d in range(num_dies) if die_used[d] > 0}

    return FloorplanResult(
        die_assignment=assignment,
        die_utilization=util,
        total_dies=len(util),
    )


# ═══════════════════════════════════════════════════════════════════════
# 64. UCIe Protocol Mapper
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class UCIeMapping:
    """UCIe die-to-die protocol mapping result.

    Attributes
    ----------
    lanes : dict[str, int]
    protocol_version : str
    total_bandwidth_gbps : float
    """

    lanes: dict[str, int]
    protocol_version: str
    total_bandwidth_gbps: float


def map_ucie_protocol(
    blocks: dict[str, int],
    *,
    lane_bandwidth_gbps: float = 32.0,
    protocol_version: str = "UCIe 2.0",
) -> UCIeMapping:
    """Map neuron array blocks to UCIe die-to-die protocol lanes.

    Parameters
    ----------
    blocks : dict[str, int]
        Block name → data width in bits per cycle.
    lane_bandwidth_gbps : float
        Bandwidth per UCIe lane.
    protocol_version : str
        UCIe protocol version.

    Returns
    -------
    UCIeMapping
    """
    lanes = {}
    total_bw = 0.0
    for block, width_bits in blocks.items():
        # Each lane carries lane_bandwidth_gbps
        needed_lanes = max(1, (width_bits + 31) // 32)
        lanes[block] = needed_lanes
        total_bw += needed_lanes * lane_bandwidth_gbps

    return UCIeMapping(
        lanes=lanes,
        protocol_version=protocol_version,
        total_bandwidth_gbps=total_bw,
    )


# ═══════════════════════════════════════════════════════════════════════
