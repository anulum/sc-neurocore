# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Independent fixed-point transitions for IR equivalence

"""Emit symbolic references for the bit-exact Python model transitions."""

from __future__ import annotations

from sc_neurocore.neurons.fixed_point_lif import FixedPointLFSR
from sc_neurocore.scpn.dcls_tent_kernel import dcls_max_forward_q88


def encoder(name: str, source: str, width: int, fraction: int, seed: int) -> str:
    """Render the Python LFSR transition and signed probability comparison."""
    taps = [bit for bit in range(16) if FixedPointLFSR(width=16, seed=1 << bit).step() & 1]
    feedback = " ^ ".join(f"{name}_rng[{bit}]" for bit in taps)
    scale = 1 << fraction
    # For a positive Q-word x, floor(x*65535 / scale) > rng iff
    # x*65535 >= (rng+1)*scale. Clamp x to the model's probability domain.
    return f"""
    reg [15:0] {name}_rng;
    wire signed [{width - 1}:0] {name}_raw = {source};
    wire [{width - 1}:0] {name}_clamped =
        {name}_raw <= 0 ? 0 :
        ($unsigned({name}_raw) > {width}'d{scale} ? {width}'d{scale} : $unsigned({name}_raw));
    wire [{width + 15}:0] {name}_scaled = {name}_clamped * 16'd65535;
    wire [{width + 16}:0] {name}_cutoff = ({{1'b0, {name}_rng}} + 17'd1) << {fraction};
    reg {name};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            {name}_rng <= 16'h{seed:04x};
            {name} <= 0;
        end else begin
            {name}_rng <= {{{name}_rng[14:0], {feedback}}};
            {name} <= {name}_scaled >= {name}_cutoff;
        end
    end
"""


def _signed_literal(value: int, width: int) -> str:
    """Render a signed constant without relying on Verilog's unsized arithmetic."""
    return f"{width}'sh{value & ((1 << width) - 1):x}"


def lif_step(
    name: str, current: str, leak: str, gain: str, noise: str, parameters: dict[str, int]
) -> str:
    """Render the Python fixed-point neuron state transition and spike output."""
    width, fraction = parameters["dw"], parameters["frac"]
    rest, reset, threshold = (
        _signed_literal(parameters[key], width) for key in ("vr", "reset", "vt")
    )
    refractory = parameters["rp"]
    return f"""
    reg {name};
    reg signed [{width - 1}:0] {name}_voltage;
    reg [31:0] {name}_hold;
    wire signed [{width - 1}:0] {name}_rest = {rest};
    wire signed [{2 * width - 1}:0] {name}_leak_product =
        ({name}_rest - {name}_voltage) * $signed({leak});
    wire signed [{2 * width - 1}:0] {name}_input_product =
        $signed({current}) * $signed({gain});
    wire signed [{width - 1}:0] {name}_leak = {name}_leak_product >>> {fraction};
    wire signed [{width - 1}:0] {name}_drive = {name}_input_product >>> {fraction};
    wire signed [{width - 1}:0] {name}_next =
        {name}_voltage + {name}_leak + {name}_drive + $signed({noise});
    wire {name}_fires = {name}_next >= {threshold};
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            {name}_voltage <= {rest};
            {name}_hold <= 0;
            {name} <= 0;
        end else begin
            {name} <= {name}_hold == 0 && {name}_fires;
            if ({name}_hold != 0) begin
                {name}_voltage <= {rest};
                {name}_hold <= {name}_hold - 32'd1;
            end else begin
                {name}_voltage <= {name}_fires ? {reset} : {name}_next;
                {name}_hold <= {name}_fires ? 32'd{refractory} : 32'd0;
            end
        end
    end
"""


def dcls_history(name: str, spike: str, offsets: tuple[int, ...]) -> tuple[str, list[str]]:
    """Encode the delayed events as a shift history, independent of the device ring buffer."""
    history_length = max(offsets)
    if history_length == 0:
        return "", [spike] * len(offsets)
    next_history = (
        spike if history_length == 1 else f"{{{name}_history[{history_length - 2}:0], {spike}}}"
    )
    source = f"""
    reg [{history_length - 1}:0] {name}_history;
    always @(posedge clk or negedge rst_n)
        if (!rst_n) {name}_history <= 0;
        else {name}_history <= {next_history};
"""
    signals = [spike if offset == 0 else f"{name}_history[{offset - 1}]" for offset in offsets]
    return source, signals


def dcls_constant(
    name: str, taps: list[str], weights: tuple[int, ...], centre: int, sigma: int
) -> str:
    """Render the Python contraction output for each symbolic tap-activity pattern."""
    cases = []
    for pattern in range(1 << len(taps)):
        active = [(pattern >> index) & 1 for index in range(len(taps))]
        # Non-positive sigma is rejected by the Python contraction and the HDL
        # exposes a zero weighted output for that invalid-domain input.
        value = dcls_max_forward_q88(active, weights, centre, sigma).output_q88 if sigma > 0 else 0
        cases.append(f"            {len(taps)}'d{pattern}: {name} = {_signed_literal(value, 16)};")
    rows = "\n".join(cases)
    signals = ", ".join(reversed(taps))
    return f"""
    reg signed [15:0] {name};
    always @* begin
        case ({{{signals}}})
{rows}
            default: {name} = 0;
        endcase
    end
"""


def dcls_symbolic(name: str, taps: list[str], weights: str, centre: str, sigma: str) -> str:
    """Lift the Q8.8 Python tent contraction to signed symbolic input words."""
    lines = [
        f"wire signed [31:0] {name}_centre = $signed({centre});",
        f"wire signed [31:0] {name}_sigma = $signed({sigma});",
        f"wire signed [31:0] {name}_divisor = {name}_sigma > 0 ? {name}_sigma : 32'sd1;",
    ]
    terms = []
    for index, active in enumerate(taps):
        tap = f"{name}_tap{index}"
        term = f"{tap}_term"
        lines.extend(
            (
                f"wire signed [31:0] {tap}_delta = 32'sd{index * 256} - {name}_centre;",
                f"wire signed [31:0] {tap}_distance = {tap}_delta < 0 ? -{tap}_delta : {tap}_delta;",
                f"wire signed [31:0] {tap}_gate = {tap}_distance < {name}_sigma ? "
                f"(({name}_sigma - {tap}_distance) <<< 8) / {name}_divisor : 32'sd0;",
                f"wire signed [15:0] {tap}_weight = {weights}[{index * 16} +: 16];",
                f"wire signed [31:0] {term} = {active} ? {tap}_weight * {tap}_gate : 32'sd0;",
            )
        )
        terms.append(term)
    lines.extend(
        (
            f"wire signed [31:0] {name}_sum = {' + '.join(terms)};",
            f"wire signed [15:0] {name} = {name}_sum > 32'sd8388352 ? 16'sh7fff : "
            f"{name}_sum < -32'sd8388608 ? 16'sh8000 : {name}_sum[23:8];",
        )
    )
    # Three products of a signed 16-bit weight and a gate <= 256 fit in i32.
    return "\n".join(lines)
