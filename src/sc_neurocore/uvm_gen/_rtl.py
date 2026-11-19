# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — UVM generator RTL contracts and module-header parser

"""Represent and parse the RTL module surface consumed by UVM generation."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum
import operator
import re
from typing import List, Optional

_MAX_SV_INT = 1 << 64
_CLOCK_NAMES = frozenset(
    ("clk", "clock", "i_clk", "clk_i", "aclk", "s_axi_aclk", "m_axi_aclk", "hclk", "pclk", "ap_clk")
)
_RESET_NAMES = frozenset(
    (
        "rst_n",
        "reset_n",
        "rst",
        "reset",
        "i_rst_n",
        "rst_ni",
        "rst_i",
        "aresetn",
        "s_axi_aresetn",
        "m_axi_aresetn",
        "hresetn",
        "presetn",
        "ap_rst_n",
        "ap_rst",
    )
)
_STRING_OR_COMMENT_RE = re.compile(r'"(?:\\.|[^"\\])*"|/\*.*?\*/|//[^\n]*', re.DOTALL)
_STRING_RE = re.compile(r'"(?:\\.|[^"\\])*"', re.DOTALL)
_COND_SCAN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|`(ifdef|ifndef|elsif)(?:\s+([A-Za-z_]\w*))?\b|`(else|endif)\b',
    re.DOTALL,
)
_IGNORED_DIRECTIVES_RE = re.compile(
    r"`(?:timescale|default_nettype|resetall|celldefine|endcelldefine)\b[^\n]*"
)
_TOKEN_RE = re.compile(
    r"""
    \s*(?:
        (\$clog2)
      | ((?:\d+)?'[sS]?[bBoOdDhH][0-9a-fA-F_]+)
      | (\d[0-9_]*)
      | ([A-Za-z_]\w*)
      | (\*\*|<<|>>|<=|>=|==|!=|&&|\|\||[?:()+\-*/%<>~!&^|])
    )
    """,
    re.VERBOSE,
)
_PRECEDENCE: tuple[tuple[str, ...], ...] = (
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<", "<=", ">", ">="),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
    ("**",),
)
_BIN_OPS: dict[str, Callable[[int, int], int]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "<": lambda a, b: int(a < b),
    "<=": lambda a, b: int(a <= b),
    ">": lambda a, b: int(a > b),
    ">=": lambda a, b: int(a >= b),
    "==": lambda a, b: int(a == b),
    "!=": lambda a, b: int(a != b),
    "&": operator.and_,
    "^": operator.xor,
    "|": operator.or_,
    "&&": lambda a, b: int(bool(a) and bool(b)),
    "||": lambda a, b: int(bool(a) or bool(b)),
}
_UNARY_OPS: dict[str, Callable[[int], int]] = {
    "+": operator.pos,
    "-": operator.neg,
    "~": operator.invert,
    "!": lambda v: int(v == 0),
}
_PARAM_TYPES = r"(?:integer|int|shortint|longint|byte|real|realtime|time|logic|reg|bit|string)"
_QUAL_SEP = r"(?:\s+|(?=\[))"
_PARAM_LEAD_RE = re.compile(
    rf"^(?:parameter|localparam)\s+"
    rf"(?:({_PARAM_TYPES}){_QUAL_SEP})?"
    rf"(?:(?:signed|unsigned){_QUAL_SEP})?"
    r"(?:\[[^\]]+\]\s*)?"
    r"(?:(?:signed|unsigned)\s+)?"
    r"([A-Za-z_]\w*)\s*=\s*(.+)$",
    re.DOTALL,
)
_PARAM_CONT_RE = re.compile(
    rf"^(?:({_PARAM_TYPES}){_QUAL_SEP})?"
    rf"(?:(?:signed|unsigned){_QUAL_SEP})?"
    r"(?:\[[^\]]+\]\s*)?"
    r"(?:(?:signed|unsigned)\s+)?"
    r"([A-Za-z_]\w*)\s*=\s*(.+)$",
    re.DOTALL,
)
_BODY_PARAM_STMT_RE = re.compile(r"\b(?:parameter|localparam)\s+([^;]+);", re.DOTALL)
_BODY_PORT_STMT_RE = re.compile(r"\b(?:input|output|inout)\b[^;]*;", re.DOTALL)
_PORT_DECL_RE = re.compile(
    r"^(input|output|inout)\s+"
    rf"(?:(logic|wire|reg){_QUAL_SEP})?"
    rf"(?:(signed|unsigned){_QUAL_SEP})?"
    r"(?:\[([^\]]+)\]\s*)?"
    r"([A-Za-z_]\w*)"
    r"((?:\s*\[[^\]]+\])*)$",
    re.DOTALL,
)
_BARE_PORT_RE = re.compile(r"^([A-Za-z_]\w*)((?:\s*\[[^\]]+\])*)$")
_IDENT_ONLY_RE = re.compile(r"^[A-Za-z_]\w*$")
_DIM_RE = re.compile(r"\[\s*([^\]]+?)\s*\]")


class PortDirection(Enum):
    """SystemVerilog port direction tokens accepted by the UVM generator."""

    INPUT = "input"
    OUTPUT = "output"
    INOUT = "inout"


class PortType(Enum):
    """SystemVerilog net/data type tokens emitted for module ports."""

    LOGIC = "logic"
    WIRE = "wire"
    REG = "reg"


@dataclass
class ModulePort:
    """Parsed RTL port metadata used by generated UVM components."""

    name: str
    direction: PortDirection
    port_type: PortType = PortType.LOGIC
    width: int = 1
    is_signed: bool = False
    is_array: bool = False
    array_size: int = 0

    @property
    def sv_decl(self) -> str:
        """Return the SystemVerilog declaration for this parsed port."""
        signed = " signed" if self.is_signed else ""
        width = f" [{self.width - 1}:0]" if self.width > 1 else ""
        arr = f" [0:{self.array_size - 1}]" if self.is_array else ""
        return f"{self.direction.value} {self.port_type.value}{signed}{width} {self.name}{arr}"

    @property
    def is_clock(self) -> bool:
        """Return whether the port name matches a supported clock convention."""
        lower = self.name.lower()
        return (
            self.width == 1
            and not self.is_array
            and (lower in _CLOCK_NAMES or lower.endswith("_aclk"))
        )

    @property
    def is_reset(self) -> bool:
        """Return whether the port name matches a supported reset convention."""
        lower = self.name.lower()
        return (
            self.width == 1
            and not self.is_array
            and (lower in _RESET_NAMES or lower.endswith("_aresetn"))
        )


@dataclass
class ModuleParam:
    """RTL module parameter."""

    name: str
    value: str
    param_type: str = "int"


def _preprocess_verilog(source: str, defines: frozenset[str]) -> str:
    cleaned = _STRING_OR_COMMENT_RE.sub(
        lambda m: m.group(0) if m.group(0).startswith('"') else " ",
        source,
    )
    out: list[str] = []
    stack: list[list[bool]] = []
    active = True
    last = 0
    for match in _COND_SCAN_RE.finditer(cleaned):
        if match.group(0).startswith('"'):
            continue
        if active:
            out.append(cleaned[last : match.start()])
        last = match.end()
        kw, macro, kw2 = match.group(1), match.group(2), match.group(3)
        if kw in ("ifdef", "ifndef"):
            if macro is None:
                raise ValueError(f"Missing macro name after `{kw}")
            cond = (macro in defines) if kw == "ifdef" else (macro not in defines)
            cur = active and cond
            stack.append([active, cond, cur, False])
            active = cur
        elif kw == "elsif":
            if not stack or stack[-1][3]:
                raise ValueError("Unexpected `elsif directive")
            if macro is None:
                raise ValueError("Missing macro name after `elsif")
            parent_act, matched, _, _ = stack[-1]
            cond = (not matched) and (macro in defines)
            cur = parent_act and cond
            stack[-1][1] = matched or cond
            stack[-1][2] = cur
            active = cur
        elif kw2 == "else":
            if not stack or stack[-1][3]:
                raise ValueError("Unexpected `else directive")
            parent_act, matched, _, _ = stack[-1]
            cur = parent_act and (not matched)
            stack[-1][1] = True
            stack[-1][2] = cur
            stack[-1][3] = True
            active = cur
        else:
            if not stack:
                raise ValueError("Unmatched `endif directive")
            stack.pop()
            active = stack[-1][2] if stack else True
    if stack:
        raise ValueError("Unterminated conditional directive")
    out.append(cleaned[last:])
    return _IGNORED_DIRECTIVES_RE.sub(" ", "".join(out))


def _reject_directives(text: str) -> None:
    bad = re.search(r"`[A-Za-z_]\w*|`", _STRING_RE.sub('""', text))
    if bad is not None:
        raise ValueError(f"Unsupported compiler directive {bad.group(0)!r}")


def _extract_balanced_parens(text: str, open_idx: int) -> tuple[str, int]:
    depth = 0
    for idx in range(open_idx, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_idx + 1 : idx], idx + 1
    raise ValueError("Unbalanced parentheses in RTL module header")


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _strip_top_level_init(text: str) -> str:
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == "=" and depth == 0:
            rhs = text[idx + 1 :].strip()
            if not rhs:
                raise ValueError(f"Empty port initialiser in {text!r}")
            return text[:idx].rstrip()
    return text


def _check_sv_bound(val: int, expr: str) -> int:
    if abs(val) > _MAX_SV_INT:
        raise ValueError(f"Constant expression magnitude exceeds 2**64 in {expr!r}")
    return val


def _parse_based_literal(token: str, expr: str) -> int:
    spec = token[token.index("'") + 1 :].lower()
    if spec.startswith("s"):
        spec = spec[1:]
    base = {"b": 2, "o": 8, "d": 10, "h": 16}[spec[0]]
    return _check_sv_bound(int(spec[1:].replace("_", ""), base), expr)


def _tokenise_expr(expr: str) -> list[tuple[str, str]]:
    stripped = expr.strip()
    if not stripped:
        raise ValueError("Empty constant expression")
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(stripped):
        match = _TOKEN_RE.match(stripped, pos)
        if match is None:
            raise ValueError(f"Unsupported token in constant expression {expr!r}")
        clog2, based, dec, ident, op = match.groups()
        if clog2:
            tokens.append(("CLOG2", clog2))
        elif based:
            tokens.append(("BASED", based))
        elif dec:
            tokens.append(("INT", dec))
        elif ident:
            tokens.append(("IDENT", ident))
        else:
            tokens.append(("OP", op))
        pos = match.end()
    return tokens


def _eval_sv_int(
    expr: str,
    symbols: dict[str, int],
    param_errors: dict[str, ValueError] | None = None,
) -> int:
    tokens = _tokenise_expr(expr)
    pos = 0

    def peek() -> tuple[str, str] | None:
        return tokens[pos] if pos < len(tokens) else None

    def consume(expected_op: str | None = None) -> tuple[str, str]:
        nonlocal pos
        tok = peek()
        if tok is None or (expected_op is not None and tok != ("OP", expected_op)):
            raise ValueError(f"Malformed constant expression {expr!r}")
        pos += 1
        return tok

    def parse_ternary() -> int:
        cond = parse_binary(0)
        if peek() == ("OP", "?"):
            consume("?")
            if_true = parse_ternary()
            consume(":")
            if_false = parse_ternary()
            return if_true if cond != 0 else if_false
        return cond

    def parse_binary(level: int) -> int:
        if level >= len(_PRECEDENCE):
            return parse_unary()
        left = parse_binary(level + 1)
        ops = _PRECEDENCE[level]
        while (tok := peek()) is not None and tok[0] == "OP" and tok[1] in ops:
            op = consume()[1]
            right = parse_binary(level + 1)
            if op == "**":
                if right < 0:
                    raise ValueError(f"Negative exponent {right} in {expr!r}")
                if right == 0:
                    left = 1
                elif abs(left) <= 1:
                    left = 0 if left == 0 else (1 if left == 1 or right % 2 == 0 else -1)
                elif right > 64 or (abs(left).bit_length() - 1) * right > 64:
                    raise ValueError(f"Constant expression magnitude exceeds 2**64 in {expr!r}")
                else:
                    left = _check_sv_bound(left**right, expr)
            elif op in ("/", "%"):
                if right == 0:
                    raise ValueError(f"Division by zero in constant expression {expr!r}")
                quot = (
                    -(abs(left) // abs(right))
                    if (left < 0) ^ (right < 0)
                    else abs(left) // abs(right)
                )
                left = _check_sv_bound(quot if op == "/" else left - quot * right, expr)
            elif op in ("<<", ">>"):
                if right < 0 or right > 64:
                    raise ValueError(f"Shift count {right} out of [0, 64] in {expr!r}")
                left = _check_sv_bound(left << right if op == "<<" else left >> right, expr)
            else:
                left = _check_sv_bound(_BIN_OPS[op](left, right), expr)
        return left

    def parse_unary() -> int:
        tok = peek()
        if tok is not None and tok[0] == "OP" and tok[1] in _UNARY_OPS:
            op = consume()[1]
            return _check_sv_bound(_UNARY_OPS[op](parse_unary()), expr)
        return parse_primary()

    def parse_primary() -> int:
        tok = consume()
        kind, text = tok
        if kind == "INT":
            return _check_sv_bound(int(text.replace("_", "")), expr)
        if kind == "BASED":
            return _parse_based_literal(text, expr)
        if kind == "IDENT":
            if peek() == ("OP", "("):
                raise ValueError(f"Unsupported constant function {text!r} in {expr!r}")
            if text in symbols:
                return symbols[text]
            if param_errors is not None and text in param_errors:
                raise param_errors[text]
            raise ValueError(f"Unresolved parameter {text!r} in {expr!r}")
        if kind == "CLOG2":
            consume("(")
            arg = parse_ternary()
            consume(")")
            if arg < 0:
                raise ValueError(f"Negative $clog2 argument {arg} in {expr!r}")
            return 0 if arg <= 1 else (arg - 1).bit_length()
        if tok == ("OP", "("):
            val = parse_ternary()
            consume(")")
            return val
        raise ValueError(f"Unexpected token {text!r} in {expr!r}")

    result = parse_ternary()
    if peek() is not None:
        raise ValueError(f"Trailing tokens in constant expression {expr!r}")
    return result


def _split_top_level_colon(text: str) -> tuple[str, str] | None:
    depth = 0
    for idx, ch in enumerate(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif ch == ":" and depth == 0:
            return text[:idx], text[idx + 1 :]
    return None


def _parse_unpacked_dims(
    dim_suffix: str,
    symbols: dict[str, int],
    param_errors: dict[str, ValueError],
) -> tuple[bool, int]:
    dims = _DIM_RE.findall(dim_suffix)
    if not dims:
        return False, 0
    total = 1
    for dim in dims:
        bounds = _split_top_level_colon(dim)
        if bounds is not None:
            hi_s, lo_s = bounds
            hi = _eval_sv_int(hi_s, symbols, param_errors)
            lo = _eval_sv_int(lo_s, symbols, param_errors)
            total *= abs(hi - lo) + 1
        else:
            size = _eval_sv_int(dim, symbols, param_errors)
            if size <= 0:
                raise ValueError(f"Non-positive unpacked dimension {dim!r}")
            total *= size
    return True, total


def _parse_param_items(
    raw_text: str,
    symbols: dict[str, int],
    param_errors: dict[str, ValueError],
    params: list[ModuleParam] | None,
    *,
    overwrite: bool,
) -> None:
    current_type: str | None = None
    for item in _split_top_level_commas(raw_text):
        entry = item.strip().rstrip(";").strip()
        if not entry:
            continue
        lead = _PARAM_LEAD_RE.match(entry)
        if lead is not None:
            current_type = lead.group(1) or "int"
            p_name, p_val = lead.group(2), " ".join(lead.group(3).split())
        elif current_type is not None and (cont := _PARAM_CONT_RE.match(entry)) is not None:
            if cont.group(1) is not None:
                current_type = cont.group(1)
            p_name, p_val = cont.group(2), " ".join(cont.group(3).split())
        else:
            raise ValueError(f"Unsupported parameter declaration: {entry!r}")
        if params is not None:
            params.append(ModuleParam(p_name, p_val, current_type))
        if overwrite or p_name not in symbols:
            try:
                symbols[p_name] = _eval_sv_int(p_val, symbols, param_errors)
                param_errors.pop(p_name, None)
            except ValueError as exc:
                param_errors.setdefault(p_name, exc)


def _parse_port_entries(
    items: list[str],
    symbols: dict[str, int],
    param_errors: dict[str, ValueError],
) -> list[ModulePort]:
    ports: list[ModulePort] = []
    last_spec: tuple[PortDirection, PortType, bool, int] | None = None
    for raw_item in items:
        stripped = raw_item.strip()
        if not stripped:
            continue
        line = _strip_top_level_init(stripped)
        pm = _PORT_DECL_RE.match(line)
        if pm is not None:
            direction = PortDirection(pm.group(1))
            port_type = PortType(pm.group(2)) if pm.group(2) else PortType.LOGIC
            is_signed = pm.group(3) == "signed"
            if pm.group(4) is not None:
                packed_bounds = _split_top_level_colon(pm.group(4))
                if packed_bounds is None:
                    raise ValueError(f"Malformed packed port range [{pm.group(4)}]")
                msb = _eval_sv_int(packed_bounds[0], symbols, param_errors)
                lsb = _eval_sv_int(packed_bounds[1], symbols, param_errors)
                width = abs(msb - lsb) + 1
            else:
                width = 1
            is_array, array_size = _parse_unpacked_dims(pm.group(6), symbols, param_errors)
            last_spec = (direction, port_type, is_signed, width)
            ports.append(
                ModulePort(
                    pm.group(5), direction, port_type, width, is_signed, is_array, array_size
                )
            )
        elif last_spec is not None and (bare_m := _BARE_PORT_RE.match(line)) is not None:
            direction, port_type, is_signed, width = last_spec
            is_array, array_size = _parse_unpacked_dims(bare_m.group(2), symbols, param_errors)
            ports.append(
                ModulePort(
                    bare_m.group(1), direction, port_type, width, is_signed, is_array, array_size
                )
            )
        else:
            raise ValueError(f"Unsupported port declaration: {stripped!r}")
    return ports


@dataclass
class RTLModule:
    """Parsed RTL module specification."""

    name: str
    ports: List[ModulePort]
    params: List[ModuleParam] = field(default_factory=list)
    is_sc_module: bool = True

    @classmethod
    def from_verilog_source(
        cls,
        source: str,
        *,
        defines: Iterable[str] = (),
    ) -> RTLModule:
        """Parse a Verilog/SystemVerilog module header and parameter widths.

        Parameters
        ----------
        source : str
            Verilog or SystemVerilog source text containing a module header.
        defines : Iterable[str], optional
            Macro names treated as defined when evaluating conditional
            compilation directives (``ifdef``, ``ifndef``, ``elsif``, ``else``,
            ``endif``). Defaults to an empty set (the synthesis view).

        Returns
        -------
        RTLModule
            Parsed module metadata with concrete packed and unpacked widths.
        """
        cleaned = _preprocess_verilog(source, frozenset(defines))
        name_match = re.search(r"\bmodule\s+([A-Za-z_]\w*)", cleaned)
        if not name_match:
            raise ValueError("No module declaration found")
        name = name_match.group(1)
        cursor = name_match.end()

        params: list[ModuleParam] = []
        symbols: dict[str, int] = {}
        param_errors: dict[str, ValueError] = {}

        hash_match = re.match(r"\s*#\s*\(", cleaned[cursor:])
        if hash_match:
            open_paren = cursor + hash_match.end() - 1
            param_text, cursor = _extract_balanced_parens(cleaned, open_paren)
            _reject_directives(param_text)
            _parse_param_items(param_text, symbols, param_errors, params, overwrite=True)

        port_open_match = re.match(r"\s*\(", cleaned[cursor:])
        port_text = ""
        if port_open_match:
            open_paren = cursor + port_open_match.end() - 1
            port_text, cursor = _extract_balanced_parens(cleaned, open_paren)
            _reject_directives(port_text)

        end_match = re.search(r"\bendmodule\b", cleaned[cursor:])
        body_text = cleaned[cursor : cursor + end_match.start()] if end_match else cleaned[cursor:]
        for stmt_match in _BODY_PARAM_STMT_RE.finditer(body_text):
            _parse_param_items(
                f"localparam {stmt_match.group(1)}",
                symbols,
                param_errors,
                None,
                overwrite=False,
            )

        raw_port_items = _split_top_level_commas(port_text) if port_text else []
        non_empty_ports = [item.strip() for item in raw_port_items if item.strip()]
        if non_empty_ports and all(_IDENT_ONLY_RE.match(item) for item in non_empty_ports):
            port_body = re.sub(
                r"\b(?:function|task)\b.*?\b(?:endfunction|endtask)\b",
                " ",
                body_text,
                flags=re.DOTALL,
            )
            body_ports: dict[str, ModulePort] = {}
            for stmt in _BODY_PORT_STMT_RE.finditer(port_body):
                stmt_body = stmt.group(0).rstrip(";").strip()
                _reject_directives(stmt_body)
                for parsed in _parse_port_entries(
                    _split_top_level_commas(stmt_body), symbols, param_errors
                ):
                    body_ports[parsed.name] = parsed
            missing = [ident for ident in non_empty_ports if ident not in body_ports]
            if missing:
                raise ValueError(f"Unsupported port declaration: {missing[0]!r}")
            ports = [body_ports[ident] for ident in non_empty_ports]
        else:
            ports = _parse_port_entries(raw_port_items, symbols, param_errors)

        return cls(name=name, ports=ports, params=params)

    @property
    def input_ports(self) -> List[ModulePort]:
        """Input ports excluding generated clock and reset controls."""
        return [
            port
            for port in self.ports
            if port.direction == PortDirection.INPUT and not port.is_clock and not port.is_reset
        ]

    @property
    def output_ports(self) -> List[ModulePort]:
        """Output ports monitored by the generated scoreboard and coverage."""
        return [port for port in self.ports if port.direction == PortDirection.OUTPUT]

    @property
    def clock_port(self) -> Optional[ModulePort]:
        """First parsed clock-like port, if the RTL module declares one."""
        return next((port for port in self.ports if port.is_clock), None)

    @property
    def reset_port(self) -> Optional[ModulePort]:
        """First parsed reset-like port, if the RTL module declares one."""
        return next((port for port in self.ports if port.is_reset), None)

    @property
    def total_input_bits(self) -> int:
        """Total data-input width excluding clock and reset ports."""
        return sum(
            port.width * (port.array_size if port.is_array else 1) for port in self.input_ports
        )

    @property
    def total_output_bits(self) -> int:
        """Total output width observed by generated verification components."""
        return sum(
            port.width * (port.array_size if port.is_array else 1) for port in self.output_ports
        )
