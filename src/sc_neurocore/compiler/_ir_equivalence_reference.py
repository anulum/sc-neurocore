# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Reference module for admitted user IR graphs

"""Build an independent reference from verified canonical IR and its declared ports."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from . import _ir_equivalence_models as models
from .equivalence_miter import MiterPort

_ASSIGN = re.compile(r"(%\d+) = sc\.([a-z_]+) (.+)")
_PORT_NAME = re.compile(r'"([A-Za-z_][A-Za-z_0-9]*)"')
_PORT = re.compile(r'"([A-Za-z_][A-Za-z_0-9]*)" (.+)')
_FIXED = re.compile(r"fixed<(\d+),(\d+)>")
_INTEGER = re.compile(r"([iu])(\d+)")
_WORD = re.compile(r"(?:[iu]64\()?(-?\d+)\)?")
_STREAM = re.compile(r"bitstream<\d+>")

STRUCTURAL_OPERATIONS = frozenset({"sc.input", "sc.output", "sc.constant"})
MODEL_OPERATIONS = frozenset({"sc.encode", "sc.lif_step", "sc.dcls_layer"})


@dataclass(frozen=True)
class Layout:
    """The scalar element representation and packed arity from an IR type."""

    width: int
    fraction: int = 0
    count: int = 1
    vector: bool = False

    @property
    def bits(self) -> int:
        """Return the width of the packed signal."""
        return self.width * self.count


@dataclass(frozen=True)
class Value:
    """One independent signal and the longest observable delay feeding it."""

    signal: str
    layout: Layout
    delay: int = 0
    observed_width: int | None = None
    computed: bool = False
    words: tuple[int, ...] = ()


@dataclass(frozen=True)
class Reference:
    """A reference module and the checked graph boundary it independently declares."""

    top: str
    verilog: str
    ports: tuple[MiterPort, ...]
    delay: int


def _layout(text: str) -> Layout:
    """Resolve the canonical signed, unsigned, rate, streaming and packed layouts."""
    if text == "bool" or _STREAM.fullmatch(text):
        return Layout(1)
    if text == "rate":
        return Layout(16, 8)
    fixed = _FIXED.fullmatch(text)
    if fixed:
        return Layout(int(fixed[1]), int(fixed[2]))
    integer = _INTEGER.fullmatch(text)
    if integer:
        return Layout(int(integer[2]))
    if text.startswith("vec<") and text.endswith(">"):
        element, arity = text[4:-1].rsplit(",", 1)
        layout = _layout(element)
        return Layout(layout.width, layout.fraction, layout.count * int(arity), vector=True)
    raise ValueError(f"unsupported IR equivalence type {text!r}")


def _constant(payload: str, layout: Layout) -> tuple[int, ...]:
    """Decode the canonical constant as raw words, truncating fractional values toward zero."""
    tokens = payload[1:-1].split(", ") if layout.vector else [payload]
    words = []
    for token in tokens:
        integer = _WORD.fullmatch(token)
        if integer:
            words.append(int(integer[1]))
        else:
            words.append(math.trunc(float(token) * (1 << layout.fraction)))
    if len(words) != layout.count:
        raise ValueError("IR equivalence constant does not match its declared shape")
    return tuple(words)


def _field_values(text: str) -> tuple[str, dict[str, str]]:
    """Split a canonical model invocation into the first operand and named fields."""
    operand, *parts = text.split(", ")
    return operand, dict(part.split("=", 1) for part in parts)


def _produce(
    keyword: str, signal: str, payload: str, values: dict[str, Value]
) -> tuple[Value, str]:
    """Emit an eligible operation with independent value and latency tracking."""
    text, type_ = payload.rsplit(" : ", 1)
    if keyword == "constant":
        layout = _layout(type_)
        words = _constant(text, layout)
        mask = (1 << layout.width) - 1
        packed = sum((word & mask) << (layout.width * index) for index, word in enumerate(words))
        return Value(signal, layout, words=words), (
            f"localparam [{layout.bits - 1}:0] {signal} = {layout.bits}'h{packed:x};"
        )
    operand, fields = _field_values(text)
    value = values[operand]
    if keyword == "encode":
        model = models.encoder(
            signal, value.signal, value.layout.width, value.layout.fraction, int(fields["seed"], 16)
        )
        return Value(signal, Layout(1), value.delay + 1, computed=True), model
    if keyword == "lif_step":
        leak, gain, noise = (values[fields[key]] for key in ("leak", "gain", "noise"))
        parameters = {key: int(fields[key]) for key in ("dw", "frac", "vr", "reset", "vt", "rp")}
        model = models.lif_step(
            signal, value.signal, leak.signal, gain.signal, noise.signal, parameters
        )
        delay = 1 + max(value.delay, leak.delay, gain.delay, noise.delay)
        return Value(signal, Layout(1), delay, parameters["dw"], computed=True), model
    return _dcls(signal, value, fields, values)


def _dcls(
    name: str, impulse: Value, fields: dict[str, str], values: dict[str, Value]
) -> tuple[Value, str]:
    """Model delays and the weighted contraction from its declared operands."""
    weights, centre, sigma = (values[fields[key]] for key in ("weights", "centre", "sigma"))
    offsets = tuple(int(index) for index in fields["offsets"][1:-1].split(";"))
    history, taps = models.dcls_history(name, impulse.signal, offsets)
    if weights.words and centre.words and sigma.words:
        model = models.dcls_constant(name, taps, weights.words, centre.words[0], sigma.words[0])
    else:
        model = models.dcls_symbolic(name, taps, weights.signal, centre.signal, sigma.signal)
    delay = max(impulse.delay + max(offsets), weights.delay, centre.delay, sigma.delay)
    return Value(name, Layout(16, 8), delay, computed=True), history + model


def _parse_port(payload: str) -> tuple[str, str]:
    """Extract a plain Verilog port name and the remainder of the IR port declaration."""
    match = _PORT.fullmatch(payload)
    if not match:
        raise ValueError(f"IR equivalence requires plain SystemVerilog port names: {payload!r}")
    return match[1], match[2]


def _declaration(port: MiterPort) -> str:
    """Render a port with the bit width independently derived from the graph."""
    return f"{port.direction} {port.declaration()}"


def build_reference(name: str, text: str) -> Reference:
    """Build the reference and its symbolic interface from native canonical IR."""
    prefix = "_reference_"
    names = _PORT_NAME.findall(text)
    while name.startswith(prefix) or any(port.startswith(prefix) for port in names):
        prefix = "_" + prefix
    top = prefix + "top"
    values: dict[str, Value] = {}
    ports = [MiterPort("clk", 1, False, "input"), MiterPort("rst_n", 1, False, "input")]
    body = []
    outputs = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("sc.output "):
            port, source = _parse_port(line.removeprefix("sc.output "))
            value = values[source]
            ports.append(
                MiterPort(port, value.observed_width or value.layout.bits, False, "output")
            )
            body.append(f"assign {port} = {value.signal};")
            outputs.append(value)
            continue
        assigned = _ASSIGN.fullmatch(line)
        if not assigned:
            if line.startswith(("sc.graph ", "sc.source ")) or line == "}":
                continue
            raise ValueError(f"cannot read canonical IR equivalence operation {line!r}")
        key, keyword, payload = assigned.groups()
        if keyword == "input":
            port, type_ = _parse_port(payload)
            layout = _layout(type_.removeprefix(": "))
            values[key] = Value(port, layout)
            ports.append(MiterPort(port, layout.bits, False, "input"))
            continue
        if f"sc.{keyword}" not in MODEL_OPERATIONS | STRUCTURAL_OPERATIONS:
            raise ValueError(f"sc.{keyword} has no independent IR equivalence reference")
        values[key], source = _produce(keyword, prefix + key[1:], payload, values)
        body.append(source)
    if not any(value.computed for value in outputs):
        raise ValueError("bounded IR equivalence needs an observable formal operation")
    if len({port.name for port in ports}) != len(ports):
        raise ValueError("bounded IR equivalence needs distinct port names")
    signature = ",\n".join(_declaration(port) for port in ports)
    source = f"module {top}(\n{signature}\n);\n" + "\n".join(body) + "\nendmodule\n"
    return Reference(top, source, tuple(ports), max(value.delay for value in outputs))
