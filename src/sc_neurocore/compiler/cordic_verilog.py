# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Clocked circular CORDIC Verilog

"""Emit a single-transaction serial circular CORDIC with a ready/valid interface."""

from __future__ import annotations

from dataclasses import dataclass

from .cordic import (
    _GUARD_BITS,
    _RANGE_GUARD_BITS,
    _ROTATION_FUNCTIONS,
    _check_function,
    _constants,
    cordic_latency,
)


@dataclass(frozen=True)
class _Mode:
    """Hold the declarations and state logic specific to one circular mode."""

    declarations: str
    accept: str
    prepare: str
    iterate: str


def _rotation(function: str, fraction: int, wide_pi: int, gain: int) -> _Mode:
    """Emit rotation with an unsigned serial remainder and signed quadrant fold."""
    wide_width = fraction + _RANGE_GUARD_BITS + 4
    reduction_shift = _RANGE_GUARD_BITS - _GUARD_BITS
    source = "y" if function == "sin" else "x"
    return _Mode(
        declarations=f"""
  localparam WIDE_WIDTH = {wide_width};
  localparam REDUCTION_STEPS = DATA_WIDTH + {_RANGE_GUARD_BITS};
  localparam signed [WIDE_WIDTH-1:0] WIDE_PI = {wide_width}'sd{wide_pi};
  localparam [WIDE_WIDTH-1:0] PERIOD = {wide_width}'d{2 * wide_pi};
  reg [REDUCTION_STEPS-1:0] input_bits;
  reg [WIDE_WIDTH-2:0] remainder;
  reg negative_angle, opposite_quadrant, phase_negative;
  reg [WIDE_WIDTH-1:0] phase_magnitude;
  reg signed [ACC_WIDTH-1:0] phase;
  wire signed [DATA_WIDTH:0] extended_input = argument0;
  wire [DATA_WIDTH:0] input_magnitude =
      extended_input < 0 ? -extended_input : extended_input;
  wire [WIDE_WIDTH-1:0] remainder_step = {{remainder, input_bits[REDUCTION_STEPS-1]}};
  wire [WIDE_WIDTH:0] subtraction =
      {{1'b0, remainder_step}} - {{1'b0, PERIOD}};
  wire [WIDE_WIDTH-1:0] reduced_step =
      subtraction[WIDE_WIDTH] ? remainder_step : subtraction[WIDE_WIDTH-1:0];
  wire [WIDE_WIDTH-1:0] unsigned_phase = {{1'b0, remainder}};
  wire upper_half = unsigned_phase >= WIDE_PI;
  wire signed [ACC_WIDTH-1:0] whole_phase = phase_magnitude >> {reduction_shift};
  wire [{reduction_shift - 1}:0] phase_tail = phase_magnitude[{reduction_shift - 1}:0];
  localparam [{reduction_shift - 1}:0] ROUND_HALF =
      {reduction_shift}'d{1 << (reduction_shift - 1)};
  wire round_phase = phase_negative ? phase_tail > ROUND_HALF : phase_tail >= ROUND_HALF;
  wire signed [ACC_WIDTH-1:0] magnitude_rounded =
      whole_phase + $signed({{1'b0, round_phase}});
  wire signed [ACC_WIDTH-1:0] internal_result = opposite_quadrant ? -{source} : {source};
""",
        accept=f"""
          input_bits <= {{input_magnitude[DATA_WIDTH-1:0], {_RANGE_GUARD_BITS}'d0}};
          remainder <= 0;
          negative_angle <= extended_input < 0;
""",
        prepare=f"""
        PREPARE: begin
          remainder <= reduced_step[WIDE_WIDTH-2:0];
          input_bits <= input_bits << 1;
          if (step == REDUCTION_STEPS - 1) begin
            step <= 0;
            state <= PHASE_MAGNITUDE;
          end else step <= step + 1'b1;
        end
        PHASE_MAGNITUDE: begin
          phase_negative <= unsigned_phase == WIDE_PI || (upper_half != negative_angle);
          phase_magnitude <= upper_half ? PERIOD - unsigned_phase : unsigned_phase;
          state <= PHASE_ROUND;
        end
        PHASE_ROUND: begin
          phase <= phase_negative ? -magnitude_rounded : magnitude_rounded;
          state <= QUADRANT;
        end
        QUADRANT: begin
          x <= {gain};
          y <= 0;
          opposite_quadrant <= phase > HALF_PI || phase < -HALF_PI;
          if (phase > HALF_PI) z <= phase - PI;
          else if (phase < -HALF_PI) z <= phase + PI;
          else z <= phase;
          state <= ITERATE;
        end
""",
        iterate="""
          if (z >= 0) begin
            x <= x - shift_y;
            y <= y + shift_x;
            z <= z - micro_angle;
          end else begin
            x <= x + shift_y;
            y <= y - shift_x;
            z <= z + micro_angle;
          end
""",
    )


def _vectoring(function: str, fraction: int, width: int) -> _Mode:
    """Emit vectoring with a bounded shared left shift and exact axial results."""
    horizontal = "argument1" if function == "atan2" else f"{width}'sd{1 << fraction}"
    internal = fraction + _GUARD_BITS
    return _Mode(
        declarations=f"""
  localparam signed [ACC_WIDTH-1:0] NORMALISED = {width}'sd{1 << internal};
  localparam NORMALISE_STEPS = {internal};
  wire signed [ACC_WIDTH-1:0] input_x = {horizontal};
  wire signed [ACC_WIDTH-1:0] input_y = argument0;
  reg negative_x, negative_y, axial;
  reg signed [ACC_WIDTH-1:0] axis_angle;
  wire signed [ACC_WIDTH-1:0] quadrant_angle = negative_x ? PI - z : z;
  wire signed [ACC_WIDTH-1:0] internal_result =
      axial ? axis_angle : (negative_y ? -quadrant_angle : quadrant_angle);
""",
        accept="""
          x <= input_x < 0 ? -input_x : input_x;
          y <= input_y < 0 ? -input_y : input_y;
          z <= 0;
          negative_x <= input_x < 0;
          negative_y <= input_y < 0;
          axial <= input_x == 0 || input_y == 0;
          if (input_y == 0) axis_angle <= input_x < 0 ? PI : 0;
          else axis_angle <= input_y < 0 ? -HALF_PI : HALF_PI;
""",
        prepare="""
        PREPARE: begin
          if (!axial && x < NORMALISED && y < NORMALISED) begin
            x <= x <<< 1;
            y <= y <<< 1;
          end
          if (step == NORMALISE_STEPS - 1) begin
            step <= 0;
            state <= ITERATE;
          end else step <= step + 1'b1;
        end
""",
        iterate="""
          if (!axial && y > 0) begin
            x <= x + shift_y;
            y <= y - shift_x;
            z <= z + micro_angle;
          end else if (!axial && y < 0) begin
            x <= x - shift_y;
            y <= y + shift_x;
            z <= z - micro_angle;
          end
""",
    )


def emit_cordic_verilog(function: str, *, data_width: int = 16, fraction: int = 8) -> str:
    """Emit a clocked ``cordic_unit`` for one circular function.

    Parameters
    ----------
    function : str
        One of sin, cos, arctan or atan2. For atan2, argument0 is the vertical
        coordinate and argument1 is the horizontal coordinate.
    data_width, fraction : int, optional
        Signed input and output geometry, as for standalone function units.

    Returns
    -------
    str
        Synthesisable Verilog. A rising edge with in_valid and in_ready
        accepts one transaction. The synchronous active-high rst aborts
        any current transaction. The result stays stable while out_valid
        is asserted until a rising edge with out_ready consumes it.
        See :func:`cordic_latency` for the exact latency in edges.

    Raises
    ------
    TypeError
        If the operation or geometry has the wrong type.
    ValueError
        If the operation or geometry is unsupported.

    """
    _check_function(function, data_width, fraction)
    constants = _constants(fraction)
    internal = fraction + _GUARD_BITS
    width = max(data_width + 3, internal + 4)
    mode = (
        _rotation(function, fraction, constants.wide_pi, constants.inverse_gain)
        if function in _ROTATION_FUNCTIONS
        else _vectoring(function, fraction, width)
    )
    second_port = (
        f"  input wire signed [{data_width - 1}:0] argument1,\n" if function == "atan2" else ""
    )
    angle_cases = "\n".join(
        f"      7'd{index}: micro_angle = {width}'sd{angle};"
        for index, angle in enumerate(constants.angles)
    )
    limit = 1 << (data_width - 1)
    latency = cordic_latency(function, data_width=data_width, fraction=fraction)
    return f"""// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Serial circular CORDIC
// Function {function}, signed {data_width}-bit words, {fraction} fractional bits.
// Latency from accepted request to valid result is {latency} clock edges.
module cordic_unit(
  input wire clk,
  input wire rst,
  input wire in_valid,
  output wire in_ready,
  input wire signed [{data_width - 1}:0] argument0,
{second_port}  output wire out_valid,
  input wire out_ready,
  output reg signed [{data_width - 1}:0] result
);
  localparam DATA_WIDTH = {data_width};
  localparam ACC_WIDTH = {width};
  localparam ITERATIONS = {len(constants.angles)};
  localparam signed [ACC_WIDTH-1:0] PI = {width}'sd{constants.pi};
  localparam signed [ACC_WIDTH-1:0] HALF_PI = {width}'sd{constants.half_pi};
  localparam signed [ACC_WIDTH-1:0] OUTPUT_MIN = -{width}'sd{limit};
  localparam signed [ACC_WIDTH-1:0] OUTPUT_MAX = {width}'sd{limit - 1};

  localparam [2:0] IDLE = 0, PREPARE = 1, QUADRANT = 2,
                   ITERATE = 3, OUTPUT_RESULT = 4, HOLD = 5,
                   PHASE_MAGNITUDE = 6, PHASE_ROUND = 7;
  reg [2:0] state;
  reg [6:0] step;
  reg signed [ACC_WIDTH-1:0] x, y, z;
  reg signed [ACC_WIDTH-1:0] micro_angle;
  wire signed [ACC_WIDTH-1:0] shift_x = x >>> step;
  wire signed [ACC_WIDTH-1:0] shift_y = y >>> step;
{mode.declarations}
  wire signed [ACC_WIDTH-1:0] rounded =
      (internal_result + {width}'sd{1 << (_GUARD_BITS - 1)}) >>> {_GUARD_BITS};

  assign in_ready = !rst && state == IDLE;
  assign out_valid = !rst && state == HOLD;

  always @* begin
    case (step)
{angle_cases}
      default: micro_angle = 0;
    endcase
  end

  always @(posedge clk) begin
    if (rst) begin
      state <= IDLE;
      step <= 0;
      result <= 0;
    end else begin
      case (state)
        IDLE: if (in_valid) begin
          step <= 0;
          state <= PREPARE;
{mode.accept}
        end
{mode.prepare}
        ITERATE: begin
{mode.iterate}
          if (step == ITERATIONS - 1) state <= OUTPUT_RESULT;
          else step <= step + 1'b1;
        end
        OUTPUT_RESULT: begin
          if (rounded < OUTPUT_MIN) result <= OUTPUT_MIN[DATA_WIDTH-1:0];
          else if (rounded > OUTPUT_MAX) result <= OUTPUT_MAX[DATA_WIDTH-1:0];
          else result <= rounded[DATA_WIDTH-1:0];
          state <= HOLD;
        end
        HOLD: if (out_ready) state <= IDLE;
        default: state <= IDLE;
      endcase
    end
  end
endmodule
"""
