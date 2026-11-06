# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Bit-true co-simulation of an emitted dense layer graph

"""Co-simulate the SystemVerilog emitted for a dense layer IR graph against Python.

The supported design is an IR graph of exactly one shape: one ``sc.input`` of type
``vec<rate,N>``, constants for the weight matrix, the leak and the gain, one
``sc.dense_forward`` reading them, and one ``sc.output`` of its spikes. The IR compiler
emits such a graph as one instance of ``sc_dense_matrix_layer``. Any other operation is
refused by name; nothing is approximated.

:func:`parse_dense_design` reads every parameter from the graph's IR text.
:func:`reference_trace` steps a bit-true Python model of the emitted hardware, cycle
by cycle, from the formulas of the RTL and the emitter; each formula names the source
lines it is taken from. :func:`run_rtl` writes a testbench, compiles the emitted top
level with the packaged library primitives using Icarus Verilog, and reads back the
same per-cycle trace. :func:`cosimulate` runs both on one stimulus and reports the
first cycle at which they differ.

A missing simulator raises :class:`SimulatorUnavailableError`; a simulator that fails
raises :class:`CosimError`. Neither is ever turned into a result.
"""

from __future__ import annotations

import hashlib
import math
import re
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sc_neurocore.hdl.resources import baseline_primitive_text
from sc_neurocore.studio.bit_true_execution import (
    NativeExecutionError,
    native_tool_version,
    run_native_command,
)

__all__ = [
    "PRIMITIVES",
    "SCHEMA",
    "TESTBENCH_MODULE",
    "CosimError",
    "CosimResult",
    "DenseDesign",
    "ScalarFormat",
    "SimulatorUnavailableError",
    "TraceRow",
    "UnsupportedDesignError",
    "cosimulate",
    "default_stimulus",
    "dotproduct_current",
    "first_difference",
    "generate_testbench",
    "lfsr_step",
    "lif_step",
    "parse_dense_design",
    "probability_threshold",
    "reference_trace",
    "run_rtl",
]

SCHEMA = "sc-neurocore.dense-cosim.v1"
"""Schema identifier of the co-simulation verdict."""

PRIMITIVES = (
    "sc_bitstream_encoder.v",
    "sc_bitstream_synapse.v",
    "sc_dotproduct_to_current.v",
    "sc_lif_neuron.v",
    "sc_dense_matrix_layer.v",
)
"""Packaged library sources the emitted dense top level instantiates, directly or not."""

TESTBENCH_MODULE = "sc_neurocore_dense_cosim_tb"
"""Name of the generated testbench module."""

# engine/src/ir/emit_sv/dense.rs lines 38-39: the emitter fixes FRACTION and
# PROBABILITY_WIDTH of every sc_dense_matrix_layer instance.
_FRACTION = 8
_PROBABILITY_WIDTH = 16
_PROBABILITY_MASK = (1 << _PROBABILITY_WIDTH) - 1
# sc_dense_matrix_layer.v lines 100 and 121: per-encoder seed strides.
_INPUT_SEED_STRIDE = 7
_WEIGHT_SEED_STRIDE = 13
# sc_lif_neuron.v lines 34-37: the layer instantiates the neuron (sc_dense_matrix_layer.v
# lines 155-158) with only DATA_WIDTH and FRACTION, so these defaults apply.
_V_REST = 0
_V_RESET = 0
_V_THRESHOLD = 1 << _FRACTION
# Stimulus words of tests/test_ir_dense_rtl.py, as signed Q8.8 rate words: below zero,
# above one, and values in between.
_STIMULUS_PATTERN = (-64, 384, 128, 256, 0, 192)
_SUPPORTED = frozenset({"input", "constant", "dense_forward", "output"})

_HEADER = re.compile(r"^sc\.graph @(?P<name>\S+) \{$")
_OPERATION = re.compile(r"^(?:%\d+\s*=\s*)?sc\.(?P<op>[a-z_]+)\b")
_INPUT = re.compile(r'^%(?P<id>\d+) = sc\.input "(?P<name>[^"]*)" : (?P<type>\S+)$')
_CONSTANT = re.compile(r"^%(?P<id>\d+) = sc\.constant (?P<value>.+) : (?P<type>\S+)$")
_DENSE = re.compile(
    r"^%(?P<id>\d+) = sc\.dense_forward %(?P<inputs>\d+), weights=%(?P<weights>\d+), "
    r"leak=%(?P<leak>\d+), gain=%(?P<gain>\d+), ni=(?P<ni>\d+), nn=(?P<nn>\d+), "
    r"len=(?P<len>\d+), dw=(?P<dw>\d+), input_seed=(?P<input_seed>\d+), "
    r"weight_seed=(?P<weight_seed>\d+), y_min=(?P<y_min>-?\d+), y_max=(?P<y_max>-?\d+) "
    r": vec<bool,\d+>$"
)
_OUTPUT = re.compile(r'^sc\.output "(?P<name>[^"]*)" %(?P<source>\d+)$')
_FIXED = re.compile(r"^fixed<(?P<width>\d+),(?P<fraction>\d+)>$")
_SIGNED = re.compile(r"^i(?P<width>\d+)$")
_VECTOR = re.compile(r"^vec<(?P<element>.+),(?P<count>\d+)>$")
_ROW = re.compile(r"^ROW (?P<words>-?\d+(?: -?\d+)*)$")


class UnsupportedDesignError(ValueError):
    """The graph is not the one dense layer shape this module models.

    Parameters
    ----------
    message : str
        What is unsupported.
    operations : Sequence[str]
        The unsupported operation keywords, such as ``sc.lif_step``, if any.
    """

    def __init__(self, message: str, operations: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.operations = tuple(operations)


class SimulatorUnavailableError(RuntimeError):
    """Icarus Verilog (``iverilog`` and ``vvp``) is not installed."""


class CosimError(RuntimeError):
    """The simulator failed to compile or run the design and its testbench."""


@dataclass(frozen=True)
class ScalarFormat:
    """Hardware word layout of one declared IR scalar type.

    Attributes
    ----------
    type : str
        The IR type as written, such as ``rate`` or ``fixed<16,8>``.
    width : int
        Bits per word.
    fraction : int
        Binary point.
    signed : bool
        Whether the word is two's complement.
    """

    type: str
    width: int
    fraction: int
    signed: bool

    def to_json(self) -> dict[str, Any]:
        """Return the layout as a JSON object.

        Returns
        -------
        dict
            ``type``, ``width``, ``fraction`` and ``signed``.
        """
        return {
            "type": self.type,
            "width": self.width,
            "fraction": self.fraction,
            "signed": self.signed,
        }


@dataclass(frozen=True)
class DenseDesign:
    """Every parameter of a supported dense layer graph, as its IR text states them.

    Attributes
    ----------
    name : str
        Graph name, which is the emitted module name.
    input_name, output_name : str
        Names of the top-level input and output ports.
    n_inputs, n_neurons : int
        Matrix geometry.
    data_width : int
        Width of the signed Q8 state, current, leak and gain words.
    stream_length : int
        Run length of the layer's step controller.
    input_seed_base, weight_seed_base : int
        Seed bases of the input and weight encoders.
    y_min, y_max : int
        Current endpoints, as raw signed Q8 words.
    input_format : ScalarFormat
        Layout of each input element (``rate``, signed Q8.8).
    weight_format : ScalarFormat
        Layout of each weight element.
    weight_words : tuple of int
        Raw signed weight words in row-major order, ``W[j][i]`` at ``j * n_inputs + i``.
    leak, gain : int
        Raw signed leak and gain words.
    leak_format, gain_format : ScalarFormat
        Layouts of the leak and gain constants.
    """

    name: str
    input_name: str
    output_name: str
    n_inputs: int
    n_neurons: int
    data_width: int
    stream_length: int
    input_seed_base: int
    weight_seed_base: int
    y_min: int
    y_max: int
    input_format: ScalarFormat
    weight_format: ScalarFormat
    weight_words: tuple[int, ...]
    leak: int
    gain: int
    leak_format: ScalarFormat
    gain_format: ScalarFormat

    @property
    def fraction(self) -> int:
        """Binary point of the state words, fixed by the emitter."""
        return _FRACTION

    @property
    def instance(self) -> str:
        """Instance name of the layer in the emitted top level.

        engine/src/ir/emit_sv.rs line 204 starts the instance counter at zero, and the
        layer is the graph's only instantiated operation.
        """
        return "u_dense_0"

    def input_seed(self, i: int) -> int:
        """Return the LFSR seed of input encoder ``i``.

        Parameters
        ----------
        i : int
            Input index.

        Returns
        -------
        int
            ``INPUT_SEED_BASE + i * 7`` (sc_dense_matrix_layer.v line 100), truncated
            to the 16-bit ``SEED_INIT`` parameter (sc_bitstream_encoder.v line 30).
        """
        return (self.input_seed_base + i * _INPUT_SEED_STRIDE) & 0xFFFF

    def weight_seed(self, j: int, i: int) -> int:
        """Return the LFSR seed of the weight encoder of neuron ``j``, input ``i``.

        Parameters
        ----------
        j : int
            Neuron index.
        i : int
            Input index.

        Returns
        -------
        int
            ``WEIGHT_SEED_BASE + j * 13 * N_INPUTS + i * 13``
            (sc_dense_matrix_layer.v line 121), truncated to 16 bits.
        """
        return (
            self.weight_seed_base
            + j * _WEIGHT_SEED_STRIDE * self.n_inputs
            + i * _WEIGHT_SEED_STRIDE
        ) & 0xFFFF

    @property
    def weight_thresholds(self) -> tuple[int, ...]:
        """Encoder thresholds of the weight words, in row-major order."""
        return tuple(probability_threshold(w, self.weight_format) for w in self.weight_words)

    def to_parameters(self) -> dict[str, Any]:
        """Return the Q formats, seeds and every other parameter as a JSON object.

        Returns
        -------
        dict
            Geometry, word formats, neuron constants, current range, controller length,
            seeds of every encoder, raw words and encoder thresholds.
        """
        return {
            "graph": self.name,
            "ports": {
                "clock": "clk",
                "reset": "rst_n (active low, asynchronous)",
                "input": self.input_name,
                "output": self.output_name,
            },
            "instance": f"sc_dense_matrix_layer {self.instance}",
            "n_inputs": self.n_inputs,
            "n_neurons": self.n_neurons,
            "q_formats": {
                "input": self.input_format.to_json(),
                "weights": self.weight_format.to_json(),
                "leak": self.leak_format.to_json(),
                "gain": self.gain_format.to_json(),
                "state": {
                    "width": self.data_width,
                    "fraction": self.fraction,
                    "signed": True,
                },
                "encoder_threshold": {
                    "width": _PROBABILITY_WIDTH,
                    "fraction": 0,
                    "signed": False,
                },
            },
            "neuron": {
                "v_rest": _V_REST,
                "v_reset": _V_RESET,
                "v_threshold": _V_THRESHOLD,
                "refractory_period": 0,
            },
            "current_range": {"y_min": self.y_min, "y_max": self.y_max},
            "leak": self.leak,
            "gain": self.gain,
            "stream_length": self.stream_length,
            "seeds": {
                "input_seed_base": self.input_seed_base,
                "weight_seed_base": self.weight_seed_base,
                "input_seed_stride": _INPUT_SEED_STRIDE,
                "weight_seed_stride": _WEIGHT_SEED_STRIDE,
                "input_encoders": [self.input_seed(i) for i in range(self.n_inputs)],
                "weight_encoders": [
                    [self.weight_seed(j, i) for i in range(self.n_inputs)]
                    for j in range(self.n_neurons)
                ],
            },
            "weights": {
                "words": list(self.weight_words),
                "encoder_thresholds": list(self.weight_thresholds),
            },
        }


@dataclass(frozen=True)
class TraceRow:
    """State of the layer after one rising clock edge.

    Attributes
    ----------
    cycle : int
        Index of the edge after reset, from zero.
    spikes : int
        The output spike vector, neuron ``j`` at bit ``j``.
    membranes : tuple of int
        Signed membrane word of every neuron.
    running : int
        The layer's ``running`` flag.
    run_done : int
        The layer's ``run_done`` flag.
    """

    cycle: int
    spikes: int
    membranes: tuple[int, ...]
    running: int
    run_done: int

    def to_json(self) -> dict[str, Any]:
        """Return the row as a JSON object.

        Returns
        -------
        dict
            ``cycle``, ``spikes``, ``membranes``, ``running`` and ``run_done``.
        """
        return {
            "cycle": self.cycle,
            "spikes": self.spikes,
            "membranes": list(self.membranes),
            "running": self.running,
            "run_done": self.run_done,
        }


@dataclass(frozen=True)
class CosimResult:
    """Reference and RTL traces of one stimulus, with the first cycle they differ at.

    Attributes
    ----------
    design : DenseDesign
        The design the reference was computed for.
    stimulus : tuple of tuple of int
        Signed Q8.8 input words applied before each edge.
    reference : tuple of TraceRow
        Python bit-true trace.
    rtl : tuple of TraceRow
        Trace printed by the simulated RTL.
    first_difference : int or None
        First differing cycle, or ``None`` when the traces are equal.
    testbench : str
        The generated testbench source.
    primitives : Mapping[str, str]
        SHA-256 of each packaged primitive source that was simulated.
    simulator : Mapping[str, str]
        First version line of ``iverilog`` and ``vvp``.
    """

    design: DenseDesign
    stimulus: tuple[tuple[int, ...], ...]
    reference: tuple[TraceRow, ...]
    rtl: tuple[TraceRow, ...]
    first_difference: int | None
    testbench: str
    primitives: Mapping[str, str] = field(default_factory=dict)
    simulator: Mapping[str, str] = field(default_factory=dict)

    @property
    def agrees(self) -> bool:
        """Whether the RTL trace equals the reference on every recorded cycle."""
        return self.first_difference is None

    def verdict(self) -> dict[str, Any]:
        """Return the outcome as a JSON object.

        Returns
        -------
        dict
            Schema, ``AGREE`` or ``DIFFER``, the number of cycles compared, the first
            differing cycle with both rows, the simulator and the primitive digests.
        """
        difference = self.first_difference
        rows: dict[str, Any] | None = None
        if difference is not None:
            rows = {
                "reference": _row_at(self.reference, difference),
                "rtl": _row_at(self.rtl, difference),
            }
        return {
            "schema": SCHEMA,
            "verdict": "AGREE" if difference is None else "DIFFER",
            "cycles": len(self.stimulus),
            "reference_rows": len(self.reference),
            "rtl_rows": len(self.rtl),
            "first_difference": difference,
            "rows_at_first_difference": rows,
            "compared": "spikes, every membrane word, running and run_done after each edge",
            "simulator": dict(self.simulator),
            "primitives_sha256": dict(self.primitives),
        }


def _row_at(rows: Sequence[TraceRow], cycle: int) -> dict[str, Any] | None:
    """Return the JSON of the row at ``cycle``, or ``None`` past the end."""
    return rows[cycle].to_json() if cycle < len(rows) else None


def _wrap(value: int, width: int) -> int:
    """Return ``value`` reduced to a signed two's-complement word of ``width`` bits."""
    value &= (1 << width) - 1
    return value - (1 << width) if value >> (width - 1) else value


def _scalar_format(text: str) -> ScalarFormat:
    """Return the layout of a scalar IR type that can carry a dense operand.

    The layouts are those of engine/src/ir/constant_words.rs lines 25-31: ``rate`` is a
    signed 16-bit word with 8 fraction bits, ``fixed<W,F>`` a signed W-bit word with F
    fraction bits, and ``iW`` a signed W-bit integer.

    Raises
    ------
    UnsupportedDesignError
        The type cannot carry a dense probability or state word.
    """
    if text == "rate":
        return ScalarFormat(text, 16, 8, True)
    fixed = _FIXED.match(text)
    if fixed is not None:
        return ScalarFormat(text, int(fixed.group("width")), int(fixed.group("fraction")), True)
    signed = _SIGNED.match(text)
    if signed is not None:
        return ScalarFormat(text, int(signed.group("width")), 0, True)
    raise UnsupportedDesignError(f"type {text} cannot carry a dense layer operand")


def _integer_word(value: int, layout: ScalarFormat) -> int:
    """Return an integer constant as a raw signed word, refusing one that does not fit.

    engine/src/ir/constant_words.rs lines 47-63: integer constants denote raw words.
    """
    limit = 1 << (layout.width - 1)
    if not -limit <= value < limit:
        raise ValueError(f"constant word {value} does not fit {layout.type}")
    return value


def _float_word(value: float, layout: ScalarFormat) -> int:
    """Quantise a float constant toward zero, as the compiler does.

    engine/src/ir/constant_words.rs lines 65-75: ``trunc(value * 2**fraction)``, which must
    lie in the signed word range.
    """
    return _integer_word(math.trunc(value * 2.0**layout.fraction), layout)


def _scalar_word(text: str, layout: ScalarFormat) -> int:
    """Return the raw word of a scalar constant as the IR parser reads its literal.

    engine/src/ir/parser.rs lines 254-286: ``i64(n)`` and ``u64(n)`` are integers; a
    literal with a ``.`` or of type ``rate`` is a float; otherwise it is an integer.
    """
    for prefix in ("i64(", "u64("):
        if text.startswith(prefix) and text.endswith(")"):
            return _integer_word(int(text[len(prefix) : -1]), layout)
    if "." in text or layout.type == "rate":
        return _float_word(float(text), layout)
    return _integer_word(int(text), layout)


def _vector_words(text: str, layout: ScalarFormat) -> tuple[int, ...]:
    """Return the raw words of a vector constant as the IR parser reads it.

    engine/src/ir/parser.rs lines 288-314: the vector is a float vector when any element
    has a ``.``, otherwise an integer vector.
    """
    elements = [part.strip() for part in text[1:-1].split(",")]
    if any("." in element for element in elements):
        return tuple(_float_word(float(element), layout) for element in elements)
    return tuple(_integer_word(int(element), layout) for element in elements)


def probability_threshold(word: int, layout: ScalarFormat) -> int:
    """Return the unsigned 16-bit encoder threshold of a signed probability word.

    engine/src/ir/emit_sv/probability.rs lines 40-53: zero for a word at or below zero;
    ``0xFFFF`` for a word at or above one when ``fraction + 1 < width``; otherwise bits
    ``[fraction +: 16]`` of ``word * 65535``.

    Parameters
    ----------
    word : int
        Raw signed word.
    layout : ScalarFormat
        Its layout.

    Returns
    -------
    int
        The ``x_value`` the encoder compares its LFSR against.
    """
    if word <= 0:
        return 0
    if layout.fraction + 1 < layout.width and word >= 1 << layout.fraction:
        return _PROBABILITY_MASK
    return ((word * 65535) >> layout.fraction) & _PROBABILITY_MASK


def lfsr_step(state: int) -> int:
    """Advance the 16-bit encoder LFSR by one clock.

    sc_bitstream_encoder.v lines 66-69 and 92: feedback from bits 15, 13, 12 and 10,
    shifted in at bit 0.

    Parameters
    ----------
    state : int
        Current register value.

    Returns
    -------
    int
        The next register value.
    """
    feedback = ((state >> 15) ^ (state >> 13) ^ (state >> 12) ^ (state >> 10)) & 1
    return ((state << 1) & 0xFFFF) | feedback


def dotproduct_current(count: int, n_inputs: int, y_min: int, y_max: int, width: int) -> int:
    """Return the current word of ``count`` coincident post-synaptic bits.

    sc_dotproduct_to_current.v lines 71-86: ``y_min + (y_max - y_min) * count / N`` with
    the product exact, the division truncated toward zero, and the sum reduced to the
    signed ``DATA_WIDTH`` word.

    Parameters
    ----------
    count : int
        Number of set post-synaptic bits.
    n_inputs : int
        ``N_INPUTS``.
    y_min, y_max : int
        Raw signed current endpoints.
    width : int
        ``DATA_WIDTH``.

    Returns
    -------
    int
        The signed current word.
    """
    product = (y_max - y_min) * count
    scaled = abs(product) // n_inputs * (1 if product >= 0 else -1)
    return _wrap(y_min + scaled, width)


def lif_step(v: int, current: int, leak: int, gain: int, width: int) -> tuple[int, int]:
    """Return the spike and the membrane word after one clock of the layer's neuron.

    sc_lif_neuron.v lines 85-93 and 111-121, with ``V_REST = V_RESET = 0``,
    ``V_THRESHOLD = 256`` and no refractory period: the leak and input products are
    formed in ``2 * DATA_WIDTH`` bits, shifted arithmetically by ``FRACTION`` and
    reduced to ``DATA_WIDTH`` bits; their sum with the membrane wraps in ``DATA_WIDTH``
    bits; a sum at or above the threshold spikes and resets.

    Parameters
    ----------
    v : int
        Signed membrane word before the edge.
    current, leak, gain : int
        Signed current, leak and gain words.
    width : int
        ``DATA_WIDTH``.

    Returns
    -------
    tuple of int
        ``(spike, membrane)`` after the edge.
    """
    dv_leak = _wrap(_wrap((_V_REST - v) * leak, 2 * width) >> _FRACTION, width)
    dv_in = _wrap(_wrap(current * gain, 2 * width) >> _FRACTION, width)
    v_next = _wrap(v + dv_leak + dv_in, width)
    if v_next >= _V_THRESHOLD:
        return 1, _V_RESET
    return 0, v_next


def parse_dense_design(text: str) -> DenseDesign:
    """Read a supported dense layer design from IR text.

    Parameters
    ----------
    text : str
        IR text as ``ScGraph.to_text`` prints it.

    Returns
    -------
    DenseDesign
        Every parameter of the design.

    Raises
    ------
    UnsupportedDesignError
        The graph has an operation other than ``sc.input``, ``sc.constant``,
        ``sc.dense_forward`` and ``sc.output`` (all are named), or not exactly the
        supported shape.
    ValueError
        A line cannot be read or a constant does not fit its type.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    header = _HEADER.match(lines[0]) if lines else None
    if header is None:
        raise ValueError("IR text does not start with an sc.graph header")
    name = header.group("name")
    unsupported = sorted(
        {
            f"sc.{match.group('op')}"
            for line in lines[1:]
            if (match := _OPERATION.match(line)) is not None and match.group("op") not in _SUPPORTED
        }
    )
    if unsupported:
        raise UnsupportedDesignError(
            f"unsupported operation(s) {', '.join(unsupported)}: the dense co-simulation "
            "supports one sc.input of vec<rate,N>, sc.constant, one sc.dense_forward "
            "and one sc.output",
            unsupported,
        )
    inputs: dict[str, tuple[str, str]] = {}
    constants: dict[str, tuple[str, str]] = {}
    dense: list[re.Match[str]] = []
    outputs: list[re.Match[str]] = []
    for line in lines[1:]:
        if line == "}":
            continue
        if (match := _INPUT.match(line)) is not None:
            inputs[match.group("id")] = (match.group("name"), match.group("type"))
        elif (match := _CONSTANT.match(line)) is not None:
            constants[match.group("id")] = (match.group("value"), match.group("type"))
        elif (match := _DENSE.match(line)) is not None:
            dense.append(match)
        elif (match := _OUTPUT.match(line)) is not None:
            outputs.append(match)
        else:
            raise ValueError(f"cannot read IR line: {line}")
    if len(inputs) != 1 or len(dense) != 1 or len(outputs) != 1:
        raise UnsupportedDesignError(
            "the dense co-simulation needs exactly one sc.input, one sc.dense_forward and "
            f"one sc.output; the graph has {len(inputs)}, {len(dense)} and {len(outputs)}"
        )
    (layer,), (output,) = dense, outputs
    input_id, (input_name, input_type) = next(iter(inputs.items()))
    n_inputs, n_neurons = int(layer.group("ni")), int(layer.group("nn"))
    if layer.group("inputs") != input_id or input_type != f"vec<rate,{n_inputs}>":
        raise UnsupportedDesignError(
            f"the layer must read the sc.input of type vec<rate,{n_inputs}>"
        )
    if output.group("source") != layer.group("id"):
        raise UnsupportedDesignError("the sc.output must be the spikes of the sc.dense_forward")
    operands = {}
    for role in ("weights", "leak", "gain"):
        if layer.group(role) not in constants:
            raise UnsupportedDesignError(f"the {role} operand must be an sc.constant")
        operands[role] = constants[layer.group(role)]
    if name in {TESTBENCH_MODULE, *(primitive[:-2] for primitive in PRIMITIVES)}:
        raise UnsupportedDesignError(f"the graph name {name} is a co-simulation module name")
    weight_value, weight_type = operands["weights"]
    vector = _VECTOR.match(weight_type)
    if vector is None or not weight_value.startswith("["):
        raise UnsupportedDesignError("the weights must be a constant vector")
    weight_format = _scalar_format(vector.group("element"))
    leak_format = _scalar_format(operands["leak"][1])
    gain_format = _scalar_format(operands["gain"][1])
    return DenseDesign(
        name=name,
        input_name=input_name,
        output_name=output.group("name"),
        n_inputs=n_inputs,
        n_neurons=n_neurons,
        data_width=int(layer.group("dw")),
        stream_length=int(layer.group("len")),
        input_seed_base=int(layer.group("input_seed")),
        weight_seed_base=int(layer.group("weight_seed")),
        y_min=int(layer.group("y_min")),
        y_max=int(layer.group("y_max")),
        input_format=_scalar_format("rate"),
        weight_format=weight_format,
        weight_words=_vector_words(weight_value, weight_format),
        leak=_scalar_word(operands["leak"][0], leak_format),
        gain=_scalar_word(operands["gain"][0], gain_format),
        leak_format=leak_format,
        gain_format=gain_format,
    )


def default_stimulus(design: DenseDesign, cycles: int = 256) -> tuple[tuple[int, ...], ...]:
    """Return the deterministic stimulus: signed Q8.8 words cycling through six levels.

    Input ``i`` holds ``(-64, 384, 128, 256, 0, 192)[(cycle // 5 + i) % 6]`` before edge
    ``cycle``: below zero, above one, and four values in between, each for five cycles,
    shifted by one level per input. This is the stimulus of ``tests/test_ir_dense_rtl.py``.

    Parameters
    ----------
    design : DenseDesign
        The design to drive.
    cycles : int, default 256
        Number of clock edges.

    Returns
    -------
    tuple of tuple of int
        One word per input for every cycle.

    Raises
    ------
    ValueError
        ``cycles`` is not positive.
    """
    if cycles < 1:
        raise ValueError("the stimulus needs at least one cycle")
    return tuple(
        tuple(
            _STIMULUS_PATTERN[(cycle // 5 + i) % len(_STIMULUS_PATTERN)]
            for i in range(design.n_inputs)
        )
        for cycle in range(cycles)
    )


def reference_trace(design: DenseDesign, stimulus: Sequence[Sequence[int]]) -> tuple[TraceRow, ...]:
    """Step the bit-true Python model of the emitted layer through a stimulus.

    Every register of the layer updates on the same edge. The neurons therefore
    integrate the encoder bits registered at the previous edge, and the encoders then
    compare their LFSR with this cycle's words (sc_bitstream_encoder.v lines 80-100).
    Reset loads every LFSR with its seed (lines 86-88, ``t_index`` tied to zero by
    sc_dense_matrix_layer.v lines 105 and 126) and clears every bit, membrane and
    controller flag.

    Parameters
    ----------
    design : DenseDesign
        The design.
    stimulus : Sequence[Sequence[int]]
        Signed Q8.8 input words applied before each edge.

    Returns
    -------
    tuple of TraceRow
        The state after every edge.

    Raises
    ------
    ValueError
        A stimulus row does not have one word per input.
    """
    n, m, width = design.n_inputs, design.n_neurons, design.data_width
    input_lfsr = [design.input_seed(i) for i in range(n)]
    weight_lfsr = [[design.weight_seed(j, i) for i in range(n)] for j in range(m)]
    weight_threshold = design.weight_thresholds
    pre_bits = [0] * n
    weight_bits = [[0] * n for _ in range(m)]
    membranes = [_V_REST] * m
    counter = running = run_done = 0
    rows: list[TraceRow] = []
    for cycle, words in enumerate(stimulus):
        if len(words) != n:
            raise ValueError(f"stimulus cycle {cycle} has {len(words)} words, not {n}")
        spikes = 0
        for j in range(m):
            # sc_bitstream_synapse.v line 24 and the AND synapses of
            # sc_dense_matrix_layer.v lines 133-139.
            count = sum(pre_bits[i] & weight_bits[j][i] for i in range(n))
            current = dotproduct_current(count, n, design.y_min, design.y_max, width)
            spike, membranes[j] = lif_step(membranes[j], current, design.leak, design.gain, width)
            spikes |= spike << j
        # sc_dense_matrix_layer.v lines 63-84 with start_pulse tied high by
        # engine/src/ir/emit_sv/dense.rs line 44.
        if not running:
            running, run_done, counter = 1, 0, 0
        elif counter == design.stream_length - 1:
            running, run_done = 0, 1
        else:
            counter += 1
        thresholds = [probability_threshold(word, design.input_format) for word in words]
        for i in range(n):
            pre_bits[i] = int(input_lfsr[i] < thresholds[i])
            input_lfsr[i] = lfsr_step(input_lfsr[i])
        for j in range(m):
            for i in range(n):
                weight_bits[j][i] = int(weight_lfsr[j][i] < weight_threshold[j * n + i])
                weight_lfsr[j][i] = lfsr_step(weight_lfsr[j][i])
        rows.append(TraceRow(cycle, spikes, tuple(membranes), running, run_done))
    return tuple(rows)


def generate_testbench(design: DenseDesign, stimulus: Sequence[Sequence[int]]) -> str:
    """Return a testbench that drives the emitted top level and prints its state.

    The bench holds reset over two clock edges, then for every cycle sets the input
    words, raises the clock and prints, after the edge, the spike vector, each
    neuron's ``v_out`` and the layer's ``running`` and ``run_done`` flags.

    Parameters
    ----------
    design : DenseDesign
        The design whose top level is driven.
    stimulus : Sequence[Sequence[int]]
        Signed Q8.8 input words for every cycle.

    Returns
    -------
    str
        SystemVerilog source of module :data:`TESTBENCH_MODULE`.
    """
    width = design.n_inputs * 16
    layer = f"dut.{design.instance}"
    probes = ", ".join(
        [f"$signed({layer}.NEURON[{j}].v_out_j)" for j in range(design.n_neurons)]
        + [f"{layer}.running", f"{layer}.run_done"]
    )
    fields = " ".join(["%0d"] * (design.n_neurons + 3))
    lines = [
        "`timescale 1ns / 1ps",
        f"module {TESTBENCH_MODULE};",
        "  reg clk = 0, rst_n = 0;",
        f"  reg [{width - 1}:0] {design.input_name};",
        f"  wire [{design.n_neurons - 1}:0] {design.output_name};",
        f"  {design.name} dut(.clk(clk), .rst_n(rst_n), .{design.input_name}"
        f"({design.input_name}), .{design.output_name}({design.output_name}));",
        "  initial begin",
        "    #1; clk = 1; #1; clk = 0; #1; clk = 1; #1; clk = 0; rst_n = 1;",
    ]
    for words in stimulus:
        packed = sum((word & 0xFFFF) << (16 * i) for i, word in enumerate(words))
        lines.append(
            f"    {design.input_name} = {width}'h{packed:0{width // 4}x}; #2; clk = 1; #2; "
            f'$display("ROW {fields}", {design.output_name}, {probes}); clk = 0; #1;'
        )
    lines += ["    $finish;", "  end", "endmodule", ""]
    return "\n".join(lines)


def _simulator() -> tuple[str, str]:
    """Return the paths of ``iverilog`` and ``vvp``.

    Raises
    ------
    SimulatorUnavailableError
        Either is not on ``PATH``.
    """
    iverilog, vvp = shutil.which("iverilog"), shutil.which("vvp")
    if iverilog is None or vvp is None:
        raise SimulatorUnavailableError(
            "Icarus Verilog (iverilog and vvp) is not installed; the dense co-simulation cannot run"
        )
    return iverilog, vvp


def _parse_rows(output: str, n_neurons: int) -> tuple[TraceRow, ...]:
    """Read the rows a testbench printed.

    Raises
    ------
    CosimError
        A row has the wrong number of fields.
    """
    rows: list[TraceRow] = []
    for line in output.splitlines():
        match = _ROW.match(line.strip())
        if match is None:
            continue
        words = [int(word) for word in match.group("words").split()]
        if len(words) != n_neurons + 3:
            raise CosimError(f"the simulator printed a malformed row: {line.strip()}")
        rows.append(TraceRow(len(rows), words[0], tuple(words[1:-2]), words[-2], words[-1]))
    return tuple(rows)


def run_rtl(
    design: DenseDesign,
    verilog: str,
    stimulus: Sequence[Sequence[int]],
    work: Path,
    *,
    timeout_s: float = 600.0,
) -> tuple[tuple[TraceRow, ...], str, dict[str, str]]:
    """Simulate the emitted top level with Icarus Verilog and read its trace.

    Parameters
    ----------
    design : DenseDesign
        The design the top level was emitted for.
    verilog : str
        The emitted SystemVerilog top level.
    stimulus : Sequence[Sequence[int]]
        Signed Q8.8 input words for every cycle.
    work : Path
        Existing directory for the sources and the compiled simulation.
    timeout_s : float, default 600.0
        Wall-clock limit for compilation and for the run, each.

    Returns
    -------
    tuple
        The RTL trace, the testbench source and the SHA-256 of each simulated primitive.

    Raises
    ------
    SimulatorUnavailableError
        ``iverilog`` or ``vvp`` is not installed.
    CosimError
        Compilation or simulation fails or prints a malformed row.
    """
    iverilog, vvp = _simulator()
    bench = generate_testbench(design, stimulus)
    (work / "design.sv").write_text(verilog, encoding="utf-8")
    (work / "testbench.sv").write_text(bench, encoding="utf-8")
    digests: dict[str, str] = {}
    for name in PRIMITIVES:
        source = baseline_primitive_text(name)
        (work / name).write_text(source, encoding="utf-8")
        digests[name] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    binary = work / "simulation.vvp"
    try:
        run_native_command(
            [
                iverilog,
                "-g2012",
                "-s",
                TESTBENCH_MODULE,
                "-o",
                str(binary),
                str(work / "design.sv"),
                str(work / "testbench.sv"),
                *(str(work / name) for name in PRIMITIVES),
            ],
            timeout_seconds=timeout_s,
        )
        completed = run_native_command([vvp, "-n", str(binary)], timeout_seconds=timeout_s)
    except NativeExecutionError as error:
        raise CosimError(str(error)) from error
    return _parse_rows(completed.stdout, design.n_neurons), bench, digests


def first_difference(
    reference: Sequence[TraceRow], rtl: Sequence[TraceRow], cycles: int
) -> int | None:
    """Return the first cycle at which two traces differ.

    Parameters
    ----------
    reference, rtl : Sequence[TraceRow]
        The two traces.
    cycles : int
        Number of cycles both should cover; a missing row is a difference.

    Returns
    -------
    int or None
        The first differing or missing cycle, or ``None`` when both hold the same
        ``cycles`` rows.
    """
    for cycle in range(max(cycles, len(reference), len(rtl))):
        if cycle >= len(reference) or cycle >= len(rtl) or reference[cycle] != rtl[cycle]:
            return cycle
    return None


def cosimulate(
    design: DenseDesign,
    verilog: str,
    stimulus: Sequence[Sequence[int]] | None = None,
    *,
    cycles: int = 256,
    timeout_s: float = 600.0,
) -> CosimResult:
    """Run the reference and the emitted RTL on one stimulus and compare them.

    Parameters
    ----------
    design : DenseDesign
        The design the reference is computed for.
    verilog : str
        The emitted SystemVerilog top level.
    stimulus : Sequence[Sequence[int]] or None, optional
        Signed Q8.8 input words per cycle; :func:`default_stimulus` when omitted.
    cycles : int, default 256
        Cycles of the default stimulus.
    timeout_s : float, default 600.0
        Wall-clock limit for compilation and for the run, each.

    Returns
    -------
    CosimResult
        Both traces and the first differing cycle.

    Raises
    ------
    SimulatorUnavailableError
        ``iverilog`` or ``vvp`` is not installed.
    CosimError
        Compilation or simulation fails.
    ValueError
        The stimulus is empty, or a row does not have one word per input.
    """
    source = default_stimulus(design, cycles=cycles) if stimulus is None else stimulus
    rows = tuple(tuple(words) for words in source)
    if not rows:
        raise ValueError("the stimulus needs at least one cycle; an empty one compares nothing")
    reference = reference_trace(design, rows)
    iverilog, vvp = _simulator()
    with tempfile.TemporaryDirectory(prefix="sc_dense_cosim_") as directory:
        rtl, bench, digests = run_rtl(design, verilog, rows, Path(directory), timeout_s=timeout_s)
    return CosimResult(
        design=design,
        stimulus=rows,
        reference=reference,
        rtl=rtl,
        first_difference=first_difference(reference, rtl, len(rows)),
        testbench=bench,
        primitives=digests,
        simulator={
            "iverilog": native_tool_version(iverilog, "iverilog"),
            "vvp": native_tool_version(vvp, "vvp"),
        },
    )
