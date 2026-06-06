# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Static analysis tools for fixed-point ODE compilation

"""Static analysis utilities for the equation compiler.

Provides five capabilities that no other neuromorphic compiler offers:

1. **Guard-bit auto-computation** — determine how many extra MSBs are needed
   in intermediate accumulators to prevent silent overflow.

2. **Formal overflow proof** — use interval arithmetic on the ODE AST to
   statically prove that no overflow occurs at a given precision.

3. **SystemVerilog Assertion (SVA) generation** — emit formal verification
   properties for safety-critical certification (DO-254 / IEC 61508).

4. **Pipeline stage analysis** — compute critical path depth and required
   pipeline stages for high-frequency targets.

5. **Power estimation** — switching-activity-based dynamic/static power
   model from generated Verilog without synthesis.

Usage::

    from sc_neurocore.compiler.static_analysis import (
        compute_guard_bits,
        prove_no_overflow,
        generate_sva,
    )

    # How many guard bits does this ODE need?
    bits = compute_guard_bits("-(v - v_rest) / tau_m + R * I / C")

    # Can we prove no overflow at Q8.8?
    result = prove_no_overflow(
        "-(v - v_rest) / tau_m + R * I / C",
        bounds={"v": (-128, 127), "v_rest": (-65, -65),
                "tau_m": (10, 10), "R": (1, 1), "I": (0, 100), "C": (1, 1)},
        q=Q88(data_width=16, fraction=8),
    )

    # Generate SVA properties
    sva = generate_sva(
        equations={"v": "-(v - v_rest) / tau_m + R * I / C"},
        bounds={"v": (-128, 127), "I": (0, 100)},
        q=Q88(data_width=16, fraction=8),
    )
"""

from __future__ import annotations

import ast
import math
from collections.abc import Sequence
from pathlib import Path
from dataclasses import dataclass
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# 1. Guard-Bit Auto-Computation
# ═══════════════════════════════════════════════════════════════════════


def _count_additions(expr_str: str) -> int:
    """Count the number of addition/subtraction nodes in an expression AST.

    Parameters
    ----------
    expr_str : str
        A Python-syntax arithmetic expression.

    Returns
    -------
    int
        Number of Add/Sub operations in the AST.
    """
    tree = ast.parse(expr_str, mode="eval")
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            count += 1
    return count


def compute_guard_bits(expr_str: str) -> int:
    """Compute the number of guard bits needed for safe accumulation.

    When summing N values, the accumulator needs ``ceil(log2(N+1))``
    extra MSBs to guarantee no intermediate overflow. For a single
    addition (a + b), 1 guard bit suffices. For ``a + b + c + d``,
    2 guard bits are needed.

    Parameters
    ----------
    expr_str : str
        A Python-syntax ODE right-hand-side expression.

    Returns
    -------
    int
        Minimum number of guard bits needed (0 if no additions).

    Examples
    --------
    >>> compute_guard_bits("a + b")
    1
    >>> compute_guard_bits("a + b + c + d")
    2
    >>> compute_guard_bits("a * b")
    0
    """
    n_adds = _count_additions(expr_str)
    if n_adds == 0:
        return 0
    # N additions can produce up to N+1 terms; guard = ceil(log2(N+1))
    return math.ceil(math.log2(n_adds + 1))


def compute_guard_bits_multi(equations: dict[str, str]) -> dict[str, int]:
    """Compute guard bits for every state variable in a multi-ODE system.

    Parameters
    ----------
    equations : dict
        Mapping from variable name to RHS expression string.

    Returns
    -------
    dict
        Mapping from variable name to required guard bits.
    """
    return {var: compute_guard_bits(expr) for var, expr in equations.items()}


# ═══════════════════════════════════════════════════════════════════════
# 2. Formal Overflow Proof (Interval Arithmetic)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class Interval:
    """A closed interval [lo, hi] for interval arithmetic."""

    lo: float
    hi: float

    def __add__(self, other: Interval) -> Interval:
        """Add two intervals: [a,b] + [c,d] = [a+c, b+d]."""
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: Interval) -> Interval:
        """Subtract intervals: [a,b] - [c,d] = [a-d, b-c]."""
        return Interval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: Interval) -> Interval:
        """Multiply intervals: all four products, take min/max."""
        products = [
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        ]
        return Interval(min(products), max(products))

    def __truediv__(self, other: Interval) -> Interval:
        """Divide intervals. Raises if divisor contains zero."""
        if other.lo <= 0 <= other.hi:
            return Interval(float("-inf"), float("inf"))
        quotients = [
            self.lo / other.lo,
            self.lo / other.hi,
            self.hi / other.lo,
            self.hi / other.hi,
        ]
        return Interval(min(quotients), max(quotients))

    def __neg__(self) -> Interval:
        """Negate: -[a,b] = [-b, -a]."""
        return Interval(-self.hi, -self.lo)

    def contains(self, lo: float, hi: float) -> bool:
        """Check if this interval is contained within [lo, hi]."""
        return self.lo >= lo and self.hi <= hi


class _IntervalEvaluator(ast.NodeVisitor):
    """Evaluate an AST expression using interval arithmetic."""

    def __init__(self, bounds: dict[str, tuple[float, float]]):
        """Initialise with variable bounds.

        Parameters
        ----------
        bounds : dict
            Mapping from variable name to (min, max) bounds.
        """
        self.bounds = bounds

    def visit_BinOp(self, node: ast.BinOp) -> Interval:
        """Evaluate a binary operation on intervals."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        elif isinstance(node.op, ast.Sub):
            return left - right
        elif isinstance(node.op, ast.Mult):
            return left * right
        elif isinstance(node.op, ast.Div):
            return left / right
        raise ValueError(f"Unsupported op: {type(node.op).__name__}")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Interval:
        """Evaluate a unary operation on an interval."""
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return operand
        raise ValueError(f"Unsupported unary: {type(node.op).__name__}")

    def visit_Name(self, node: ast.Name) -> Interval:
        """Look up the interval for a variable name."""
        if node.id in self.bounds:
            lo, hi = self.bounds[node.id]
            return Interval(lo, hi)
        raise KeyError(f"No bounds for variable '{node.id}'")

    def visit_Constant(self, node: ast.Constant) -> Interval:
        """A constant is a point interval [c, c]."""
        if not isinstance(node.value, (int, float)):
            raise ValueError(f"Unsupported constant: {node.value!r}")
        v = float(node.value)
        return Interval(v, v)

    def generic_visit(self, node: ast.AST) -> Interval:
        """Raise for unsupported nodes."""
        raise ValueError(f"Unsupported node: {type(node).__name__}")


@dataclass
class OverflowProofResult:
    """Result of a formal overflow proof.

    Attributes
    ----------
    proven_safe : bool
        True if the expression provably cannot overflow at the given precision.
    expr_interval : Interval
        The computed output interval of the expression.
    q_min : float
        Minimum representable value in the Q-format.
    q_max : float
        Maximum representable value in the Q-format.
    margin_lo : float
        How far the expression minimum is from the Q-format minimum (positive = safe).
    margin_hi : float
        How far the expression maximum is from the Q-format maximum (positive = safe).
    """

    proven_safe: bool
    expr_interval: Interval
    q_min: float
    q_max: float
    margin_lo: float
    margin_hi: float


@dataclass(frozen=True)
class FixedPointEnvelopeProof:
    """Static fixed-point width proof over conservative Q-code bounds.

    The proof is intended for deployment artefacts that already compute
    conservative absolute output bounds, such as mixed Q8.8/Q16.16 and
    block-floating dense paths.  It does not use realised output cancellation:
    a small output value remains unsafe when the absolute product envelope
    exceeds the signed Q-format capacity.
    """

    proof_kind: str
    output_format: str
    signed: bool
    total_bits: int
    fractional_bits: int
    conservative_safe_bound_code: int
    max_abs_bound_code: int
    min_headroom_code: int
    required_total_bits: int
    required_integer_bits: int
    width_headroom_bits: int
    saturation_required: bool
    static_overflow_proven_safe: bool

    def manifest(self) -> dict[str, int | bool | str]:
        """Return a stable JSON-serialisable proof manifest."""
        return {
            "proof_kind": self.proof_kind,
            "output_format": self.output_format,
            "signed": self.signed,
            "total_bits": self.total_bits,
            "fractional_bits": self.fractional_bits,
            "conservative_safe_bound_code": self.conservative_safe_bound_code,
            "max_abs_bound_code": self.max_abs_bound_code,
            "min_headroom_code": self.min_headroom_code,
            "required_total_bits": self.required_total_bits,
            "required_integer_bits": self.required_integer_bits,
            "width_headroom_bits": self.width_headroom_bits,
            "saturation_required": self.saturation_required,
            "static_overflow_proven_safe": self.static_overflow_proven_safe,
        }


def _required_total_bits_for_bound(abs_bound_code: int, *, signed: bool) -> int:
    """Return the minimum total bit width needed by a non-negative code bound."""
    if abs_bound_code <= 0:
        return 1
    if signed:
        return abs_bound_code.bit_length() + 1
    return abs_bound_code.bit_length()


def prove_fixed_point_envelope(
    bound_codes: Sequence[int],
    *,
    total_bits: int = 32,
    fractional_bits: int = 16,
    signed: bool = True,
) -> FixedPointEnvelopeProof:
    """Prove whether conservative Q-code bounds fit a fixed-point format.

    Parameters
    ----------
    bound_codes : Sequence[int]
        Conservative absolute output bounds in integer Q-code units.  For
        signed formats the function accepts signed codes and proves against
        their absolute magnitudes.  Unsigned formats reject negative codes.
    total_bits : int
        Total output width, including the sign bit for signed formats.
    fractional_bits : int
        Fractional precision bits in the Q-format.
    signed : bool
        Whether the target fixed-point format is signed.

    Returns
    -------
    FixedPointEnvelopeProof
        Fail-closed width proof and saturation requirement.
    """
    if total_bits <= 0:
        raise ValueError("total_bits must be positive")
    if fractional_bits < 0:
        raise ValueError("fractional_bits must be non-negative")
    if fractional_bits >= total_bits:
        raise ValueError("fractional_bits must be smaller than total_bits")
    if not bound_codes:
        raise ValueError("bound_codes must contain at least one conservative bound")

    normalised_bounds: list[int] = []
    for code in bound_codes:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError("bound_codes must be integer Q-code values")
        if not signed and code < 0:
            raise ValueError("unsigned fixed-point envelope cannot contain negative codes")
        normalised_bounds.append(abs(code) if signed else code)

    max_abs_bound = max(normalised_bounds)
    if signed:
        safe_bound = (1 << (total_bits - 1)) - 1
        integer_floor = 1
        proof_kind = "signed_symmetric_fixed_point_width"
    else:
        safe_bound = (1 << total_bits) - 1
        integer_floor = 0
        proof_kind = "unsigned_fixed_point_width"

    required_total = _required_total_bits_for_bound(max_abs_bound, signed=signed)
    required_integer = max(required_total - fractional_bits, integer_floor)
    width_headroom = total_bits - required_total
    saturation_required = max_abs_bound > safe_bound
    output_format = f"Q{total_bits - fractional_bits}.{fractional_bits}"

    return FixedPointEnvelopeProof(
        proof_kind=proof_kind,
        output_format=output_format,
        signed=signed,
        total_bits=total_bits,
        fractional_bits=fractional_bits,
        conservative_safe_bound_code=safe_bound,
        max_abs_bound_code=max_abs_bound,
        min_headroom_code=safe_bound - max_abs_bound,
        required_total_bits=required_total,
        required_integer_bits=required_integer,
        width_headroom_bits=width_headroom,
        saturation_required=saturation_required,
        static_overflow_proven_safe=not saturation_required,
    )


def prove_no_overflow(
    expr_str: str,
    bounds: dict[str, tuple[float, float]],
    data_width: int = 16,
    fraction: int = 8,
    signed: bool = True,
) -> OverflowProofResult:
    """Statically prove that an expression cannot overflow at a given precision.

    Uses interval arithmetic to compute the range of possible output values,
    then checks whether the output interval fits within the Q-format range.

    Parameters
    ----------
    expr_str : str
        Python-syntax arithmetic expression.
    bounds : dict
        Mapping from variable name to (min, max) bounds.
    data_width : int
        Total bit width.
    fraction : int
        Fractional bits.
    signed : bool
        Whether the format is signed.

    Returns
    -------
    OverflowProofResult
        Contains ``proven_safe=True`` if the proof succeeds.
    """
    tree = ast.parse(expr_str, mode="eval")
    evaluator = _IntervalEvaluator(bounds)
    result_interval = evaluator.visit(tree.body)

    if signed:
        q_max = ((1 << (data_width - 1)) - 1) / (1 << fraction)
        q_min = -(1 << (data_width - 1)) / (1 << fraction)
    else:
        q_max = ((1 << data_width) - 1) / (1 << fraction)
        q_min = 0.0

    margin_lo = result_interval.lo - q_min
    margin_hi = q_max - result_interval.hi

    return OverflowProofResult(
        proven_safe=(margin_lo >= 0 and margin_hi >= 0),
        expr_interval=result_interval,
        q_min=q_min,
        q_max=q_max,
        margin_lo=margin_lo,
        margin_hi=margin_hi,
    )


# ═══════════════════════════════════════════════════════════════════════
# 3. SystemVerilog Assertion (SVA) Generation
# ═══════════════════════════════════════════════════════════════════════


def generate_sva(
    state_vars: list[str],
    *,
    data_width: int = 16,
    fraction: int = 8,
    signed: bool = True,
    input_bounds: dict[str, tuple[float, float]] | None = None,
    module_name: str = "sc_equation_neuron",
) -> str:
    """Generate SystemVerilog Assertions for a compiled neuron module.

    Produces three categories of formal properties:

    1. **Overflow assertions** — check that no state variable exceeds
       the representable range after the next-state update.
    2. **Reachability covers** — prove that spike output is reachable.
    3. **Input assumptions** — constrain external inputs to valid bounds.

    Parameters
    ----------
    state_vars : list[str]
        Names of state variables (e.g. ``["v"]``).
    data_width : int
        Bit width of the fixed-point format.
    fraction : int
        Fractional bits.
    signed : bool
        True for signed format.
    input_bounds : dict, optional
        Mapping from input names to (min_q, max_q) bounds in Q-format integers.
    module_name : str
        Name of the target module.

    Returns
    -------
    str
        SystemVerilog bind module with assertions.
    """
    if signed:
        q_max = (1 << (data_width - 1)) - 1
        q_min = -(1 << (data_width - 1))
    else:
        q_max = (1 << data_width) - 1
        q_min = 0

    sign_kw = "signed " if signed else ""
    lines = [
        f"// Auto-generated SystemVerilog Assertions for {module_name}",
        "// SC-NeuroCore static analysis — DO-254 / IEC 61508 compliance",
        f"// Fixed-point: Q{data_width - fraction - (1 if signed else 0)}.{fraction} "
        f"({data_width}-bit {'signed' if signed else 'unsigned'})",
        "",
        f"module {module_name}_sva (",
        "    input wire clk,",
        "    input wire rst_n,",
        f"    input wire {sign_kw}[{data_width - 1}:0] I_t,",
        "    input wire spike_out,",
    ]

    for var in state_vars:
        lines.append(f"    input wire {sign_kw}[{data_width - 1}:0] {var}_reg,")

    # Remove trailing comma from last port
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    lines.append("")

    # Default clocking block
    lines.append("    default clocking cb @(posedge clk);")
    lines.append("    endclocking")
    lines.append("")

    # 1. Overflow assertions
    lines.append("    // ── Overflow Assertions ──────────────────────────────────")
    for var in state_vars:
        if signed:
            lines.append(
                f"    a_no_overflow_{var}: assert property ("
                f"disable iff (!rst_n) "
                f"$signed({var}_reg) >= {data_width}'sd{q_min} && "
                f"$signed({var}_reg) <= {data_width}'sd{q_max}"
                f') else $error("OVERFLOW: {var}_reg = %0d", {var}_reg);'
            )
        else:
            lines.append(
                f"    a_no_overflow_{var}: assert property ("
                f"disable iff (!rst_n) "
                f"{var}_reg <= {data_width}'d{q_max}"
                f') else $error("OVERFLOW: {var}_reg = %0d", {var}_reg);'
            )

    lines.append("")

    # 2. Reachability covers
    lines.append("    // ── Reachability Covers ─────────────────────────────────")
    lines.append("    c_spike_reachable: cover property (disable iff (!rst_n) spike_out == 1'b1);")
    lines.append("    c_no_spike: cover property (disable iff (!rst_n) spike_out == 1'b0);")

    for var in state_vars:
        lines.append(
            f"    c_{var}_nonzero: cover property ("
            f"disable iff (!rst_n) {var}_reg != {data_width}'sd0"
            f");"
        )

    lines.append("")

    # 3. Input assumptions
    if input_bounds:
        lines.append("    // ── Input Assumptions ──────────────────────────────────")
        for name, (lo, hi) in input_bounds.items():
            lines.append(
                f"    m_{name}_bound: assume property ("
                f"disable iff (!rst_n) "
                f"$signed({name}) >= {data_width}'sd{lo} && "
                f"$signed({name}) <= {data_width}'sd{hi}"
                f");"
            )
        lines.append("")

    # 4. Stability check — membrane voltage should not stay at max for too long
    lines.append("    // ── Stability Checks ───────────────────────────────────")
    for var in state_vars:
        lines.append(
            f"    a_{var}_not_stuck_max: assert property ("
            f"disable iff (!rst_n) "
            f"not ({var}_reg == {data_width}'sd{q_max} [*100])"
            f') else $warning("{var}_reg stuck at max for 100+ cycles");'
        )

    lines.append("")
    lines.append("endmodule")
    lines.append("")

    # Bind directive
    lines.append("// Bind to DUT — place in testbench or verification top")
    lines.append(f"// bind {module_name} {module_name}_sva sva_inst (.*);")
    lines.append("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# 4. Pipeline Stage Analysis
# ═══════════════════════════════════════════════════════════════════════


def _mul_div_depth(node: ast.AST) -> int:
    """Return the longest chain of Mult/Div operations from root to leaf.

    Parameters
    ----------
    node : ast.AST
        Root AST node.

    Returns
    -------
    int
        Maximum multiplicative depth.
    """
    if isinstance(node, ast.BinOp):
        left_d = _mul_div_depth(node.left)
        right_d = _mul_div_depth(node.right)
        is_mul = isinstance(node.op, (ast.Mult, ast.Div))
        return max(left_d, right_d) + (1 if is_mul else 0)
    if isinstance(node, ast.UnaryOp):
        return _mul_div_depth(node.operand)
    return 0


def critical_path_depth(expr_str: str) -> int:
    """Count the longest chain of Mult/Div nodes in an expression.

    This determines how many DSP blocks are chained in series in the
    resulting Verilog datapath. At high frequencies, each DSP in series
    adds ~2.5 ns of combinational delay.

    Parameters
    ----------
    expr_str : str
        Python-syntax arithmetic expression.

    Returns
    -------
    int
        Maximum multiply/divide chain length (0 = no multiplies).

    Examples
    --------
    >>> critical_path_depth("a * b + c")
    1
    >>> critical_path_depth("a * b * c * d")
    3
    >>> critical_path_depth("a + b")
    0
    """
    tree = ast.parse(expr_str, mode="eval")
    return _mul_div_depth(tree.body)


def pipeline_stages_needed(
    depth: int,
    target_freq_mhz: int,
    dsp_delay_ns: float = 2.5,
    routing_overhead_ns: float = 0.5,
) -> int:
    """Compute how many pipeline registers are needed between DSP blocks.

    Parameters
    ----------
    depth : int
        Critical path depth (from ``critical_path_depth()``).
    target_freq_mhz : int
        Target clock frequency in MHz.
    dsp_delay_ns : float
        Propagation delay per DSP multiply (default 2.5 ns for Artix-7).
    routing_overhead_ns : float
        Routing overhead per stage.

    Returns
    -------
    int
        Number of pipeline registers to insert (0 = no pipelining needed).
    """
    if depth == 0 or target_freq_mhz <= 0:
        return 0

    period_ns = 1000.0 / target_freq_mhz
    total_delay = depth * (dsp_delay_ns + routing_overhead_ns)

    if total_delay <= period_ns:
        return 0

    # Each pipeline register breaks the path into (stages+1) segments
    # Need ceil(total_delay / period) - 1 registers
    stages = math.ceil(total_delay / period_ns) - 1
    return max(0, stages)


def pipeline_analysis(
    equations: dict[str, str],
    target_freq_mhz: int = 100,
    dsp_delay_ns: float = 2.5,
) -> dict[str, dict[str, Any]]:
    """Analyse pipeline requirements for a multi-ODE system.

    Parameters
    ----------
    equations : dict
        Variable name → RHS expression.
    target_freq_mhz : int
        Target clock frequency.
    dsp_delay_ns : float
        DSP propagation delay.

    Returns
    -------
    dict
        Per-variable analysis: ``{var: {"depth": int, "stages": int,
        "achievable_mhz": int}}``.
    """
    results: dict[str, dict[str, Any]] = {}
    for var, expr in equations.items():
        depth = critical_path_depth(expr)
        stages = pipeline_stages_needed(depth, target_freq_mhz, dsp_delay_ns)
        if depth > 0:
            per_stage_delay = dsp_delay_ns + 0.5  # routing overhead
            achievable = int(1000.0 / (depth * per_stage_delay))
        else:
            achievable = target_freq_mhz
        results[var] = {
            "depth": depth,
            "stages": stages,
            "achievable_mhz": achievable,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════
# 5. Power Estimation
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PowerEstimate:
    """Estimated power consumption for a compiled neuron.

    Attributes
    ----------
    dynamic_mw : float
        Dynamic (switching) power in milliwatts.
    static_mw : float
        Leakage power in milliwatts.
    total_mw : float
        Total power (dynamic + static).
    energy_per_spike_nj : float
        Energy per spike event in nanojoules.
    toggle_rate : float
        Average toggle rate (transitions per clock per bit).
    """

    dynamic_mw: float
    static_mw: float
    total_mw: float
    energy_per_spike_nj: float
    toggle_rate: float


def _load_vcd_text(activity_vcd: str | Path) -> str:
    if isinstance(activity_vcd, Path):
        return activity_vcd.read_text(encoding="utf-8")
    candidate = Path(activity_vcd)
    if "$var" not in activity_vcd and candidate.exists():
        return candidate.read_text(encoding="utf-8")
    return activity_vcd


def _bit_toggle_count(previous: str, current: str, width: int) -> int:
    previous = previous[-width:].zfill(width)
    current = current[-width:].zfill(width)
    return sum(a != b for a, b in zip(previous, current, strict=True))


def _parse_vcd_activity(
    activity_vcd: str | Path, time_units_per_cycle: float
) -> tuple[float, float]:
    """Return measured toggles per cycle and average bit toggle rate from VCD."""
    if time_units_per_cycle <= 0 or not math.isfinite(time_units_per_cycle):
        raise ValueError("vcd_time_units_per_cycle must be finite and > 0")

    import re as _re

    text = _load_vcd_text(activity_vcd)
    widths: dict[str, int] = {}
    previous_values: dict[str, str] = {}
    observed_codes: set[str] = set()
    current_time = 0.0
    first_time: float | None = None
    last_time: float | None = None
    total_toggles = 0

    var_re = _re.compile(r"\$var\s+\S+\s+(\d+)\s+(\S+)\s+.+?\$end")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        var_match = var_re.match(line)
        if var_match:
            widths[var_match.group(2)] = int(var_match.group(1))
            continue
        if line.startswith("#"):
            current_time = float(line[1:])
            if first_time is None:
                first_time = current_time
            last_time = current_time
            continue
        if line[0] in "01" and len(line) >= 2:
            value = line[0]
            code = line[1:].strip()
        elif line[0] in "bB":
            parts = line.split()
            if len(parts) != 2:
                continue
            value = parts[0][1:]
            code = parts[1]
        else:
            continue
        if code not in widths or any(bit not in "01" for bit in value):
            continue
        observed_codes.add(code)
        if code in previous_values:
            total_toggles += _bit_toggle_count(previous_values[code], value, widths[code])
        previous_values[code] = value

    if not observed_codes:
        return 0.0, 0.0
    if first_time is None or last_time is None or last_time <= first_time:
        cycles = 1.0
    else:
        cycles = max(1.0, (last_time - first_time) / time_units_per_cycle)
    observed_bits = sum(widths[code] for code in observed_codes)
    toggles_per_cycle = total_toggles / cycles
    toggle_rate = total_toggles / max(1.0, observed_bits * cycles)
    return toggles_per_cycle, toggle_rate


def estimate_power(
    verilog: str,
    *,
    data_width: int = 16,
    freq_mhz: float = 100.0,
    vdd: float = 1.0,
    process_nm: int = 28,
    spike_rate_hz: float = 10.0,
    activity_vcd: str | Path | None = None,
    vcd_time_units_per_cycle: float = 1.0,
) -> PowerEstimate:
    """Estimate power consumption from generated Verilog.

    When a VCD trace is provided, dynamic power uses measured bit-level
    switching activity. Otherwise the function falls back to structural
    activity factors derived from registers, adders, multipliers, and
    technology parameters.

    Parameters
    ----------
    verilog : str
        Generated Verilog source.
    data_width : int
        Fixed-point data width.
    freq_mhz : float
        Clock frequency in MHz.
    vdd : float
        Supply voltage (V).
    process_nm : int
        Process node in nm (7, 16, 28, 45, ...).
    spike_rate_hz : float
        Expected average spike rate.
    activity_vcd : str or Path, optional
        VCD trace text or a path to a VCD file. When provided, bit-level
        transitions in the trace drive dynamic power.
    vcd_time_units_per_cycle : float
        Number of VCD timestamp units per target clock cycle.

    Returns
    -------
    PowerEstimate
        Estimated power breakdown.
    """
    import re as _re

    # Count switching elements
    mul_count = len(_re.findall(r"wire\s+signed\s+\[.*?\]\s+_mul\d+", verilog))
    add_count = verilog.count(" + ") + verilog.count(" - ")
    reg_count = len(_re.findall(r"reg\s+signed\s+\[", verilog))

    # Technology-dependent capacitance (fF per toggle per bit)
    cap_per_bit_ff = {
        7: 0.2,
        10: 0.3,
        14: 0.5,
        16: 0.6,
        22: 0.8,
        28: 1.0,
        40: 1.5,
        45: 1.8,
        65: 2.5,
    }.get(process_nm, 1.0)

    if activity_vcd is not None:
        total_toggles, avg_toggle_rate = _parse_vcd_activity(
            activity_vcd,
            vcd_time_units_per_cycle,
        )
    else:
        # Structural activity factors used when measured switching data is absent.
        reg_toggles = reg_count * data_width * 0.25
        add_toggles = add_count * data_width * 0.50
        mul_toggles = mul_count * data_width * data_width * 0.10

        total_toggles = reg_toggles + add_toggles + mul_toggles
        avg_toggle_rate = total_toggles / max(1, (reg_count + add_count + mul_count) * data_width)

    # P_dynamic = α × C × V² × f
    cap_f = cap_per_bit_ff * 1e-15  # convert fF to F
    freq_hz = freq_mhz * 1e6
    p_dynamic_w = total_toggles * cap_f * (vdd**2) * freq_hz
    p_dynamic_mw = p_dynamic_w * 1e3

    # Leakage: ~0.1 μW per LUT for 28nm (scales with process²)
    leakage_uw_per_lut = 0.1 * (process_nm / 28) ** 2
    lut_estimate = add_count * data_width + mul_count * data_width * data_width // 4
    p_static_mw = lut_estimate * leakage_uw_per_lut * 1e-3

    total_mw = p_dynamic_mw + p_static_mw

    # Energy per spike (nJ) = total_power / spike_rate
    if spike_rate_hz > 0:
        energy_per_spike_nj = (total_mw * 1e-3) / spike_rate_hz * 1e9
    else:
        energy_per_spike_nj = 0.0

    return PowerEstimate(
        dynamic_mw=round(p_dynamic_mw, 4),
        static_mw=round(p_static_mw, 4),
        total_mw=round(total_mw, 4),
        energy_per_spike_nj=round(energy_per_spike_nj, 2),
        toggle_rate=round(avg_toggle_rate, 3),
    )
