# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Research AER wrapper and dense-layer HDL emission helpers

"""Emit clocked AER wrappers and ``Dense`` layer bindings.

A ``Dense`` layer whose ``weights`` is a 2-D ``n_neurons x n_inputs`` matrix binds
``sc_dense_matrix_layer``: one weight row per neuron, each with its own encoders, seeded from
``input_seed_base`` and ``weight_seed_base``. A 1-D ``weights`` (or one ``weight_fp``) binds
``sc_dense_layer_core``, a shared-weight population: one dot product drives every neuron with
the same current, so its ``n_neurons`` spike bits are one signal. That core has fixed encoder
seeds, so seed parameters are refused on it rather than ignored.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, log2
from numbers import Integral
from typing import Any

from ._ident import sanitize_ident


def require_positive_int(value: object, name: str) -> int:
    """Validate and convert a positive integer parameter.

    Parameters
    ----------
    value : object
        Candidate integer value (booleans are rejected).
    name : str
        Field label included in any validation error.

    Returns
    -------
    int
        Validated positive integer.
    """
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _require_non_bool_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _format_signed_word(value: object, data_width: int, name: str) -> str:
    ival = _require_non_bool_int(value, name)
    lo = -(1 << (data_width - 1))
    hi = (1 << (data_width - 1)) - 1
    if not lo <= ival <= hi:
        raise ValueError(f"{name} ({ival}) out of signed {data_width}-bit range [{lo}, {hi}]")
    masked = ival & ((1 << data_width) - 1)
    hex_digits = (data_width + 3) // 4
    return f"{data_width}'h{masked:0{hex_digits}X}"


def _format_unsigned_word(value: object, data_width: int, name: str) -> str:
    ival = _require_non_bool_int(value, name)
    hi = (1 << data_width) - 1
    if not 0 <= ival <= hi:
        raise ValueError(f"{name} ({ival}) out of unsigned {data_width}-bit range [0, {hi}]")
    hex_digits = (data_width + 3) // 4
    return f"{data_width}'h{ival:0{hex_digits}X}"


def _require_params_mapping(value: object, layer_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Dense layer '{layer_name}' params must be a mapping")
    return value


_SEED_KEYS = ("input_seed_base", "weight_seed_base")
_DEFAULT_SEEDS = {"input_seed_base": 0xACE1, "weight_seed_base": 0xBEEF}


def dense_weight_matrix(params: Mapping[str, object]) -> list[list[object]] | None:
    """Return the weight rows of a 2-D ``weights`` matrix, or ``None`` for the shared-core forms.

    Parameters
    ----------
    params : Mapping[str, object]
        ``Dense`` layer parameters.

    Returns
    -------
    list[list[object]] or None
        One row per neuron when ``weights`` is a list or tuple of lists or tuples; ``None`` when
        ``weights`` is absent or 1-D, which bind ``sc_dense_layer_core``.

    Raises
    ------
    ValueError
        If ``weights`` mixes rows and scalars.
    """
    raw = params.get("weights")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    rows: list[list[object]] = []
    scalars = 0
    for item in raw:
        if isinstance(item, (list, tuple)):
            rows.append(list(item))
        else:
            scalars += 1
    if not rows:
        return None
    if scalars:
        raise ValueError("Dense weights must be all rows (2-D) or all scalars (1-D), not a mix")
    return rows


def validate_dense_layer_params(layer_name: str, params: object) -> None:
    """Validate ``Dense`` layer configuration for its HDL binding.

    Every parameter is either honoured by the module the layer binds or refused; none is
    accepted and then ignored.

    - Both bindings honour ``n_neurons``, ``input_width``, ``output_width``, ``data_width``,
      ``stream_len``, ``y_min_fp``, ``y_max_fp``, ``cfg_leak`` and ``cfg_gain``: each reaches a
      parameter or port of the emitted instance.
    - ``sc_dense_layer_core`` (1-D ``weights``, one ``weight_fp``, or neither) also honours
      ``weights`` and ``weight_fp``, and refuses ``input_seed_base`` and ``weight_seed_base``,
      because its encoder seeds are fixed inside the module.
    - ``sc_dense_matrix_layer`` (2-D ``weights``) also honours ``input_seed_base`` and
      ``weight_seed_base`` as ``INPUT_SEED_BASE`` and ``WEIGHT_SEED_BASE``.

    Parameters
    ----------
    layer_name : str
        Sanitised Verilog layer identifier.
    params : object
        Layer parameter dictionary.

    Raises
    ------
    ValueError
        If a parameter is missing, not an integer, out of range or not honoured by the bound
        module.
    TypeError
        If ``params`` is not a mapping.
    """
    checked = _require_params_mapping(params, layer_name)
    if "n_neurons" not in checked:
        raise ValueError(f"Dense layer '{layer_name}' requires n_neurons")
    n_neurons = require_positive_int(checked["n_neurons"], f"Dense layer '{layer_name}' n_neurons")
    if "input_width" in checked:
        require_positive_int(checked["input_width"], f"Dense layer '{layer_name}' input_width")
    if "output_width" in checked:
        out_w = require_positive_int(
            checked["output_width"], f"Dense layer '{layer_name}' output_width"
        )
        if out_w != n_neurons:
            raise ValueError(
                f"Dense layer '{layer_name}' output_width ({out_w}) "
                f"must equal n_neurons ({n_neurons})"
            )
    data_width = require_positive_int(
        checked.get("data_width", 16), f"Dense layer '{layer_name}' data_width"
    )
    if data_width < 16:
        raise ValueError(f"Dense layer '{layer_name}' data_width ({data_width}) must be >= 16")
    stream_len = require_positive_int(
        checked.get("stream_len", 256), f"Dense layer '{layer_name}' stream_len"
    )
    if stream_len > 0xFFFF_FFFF:
        raise ValueError(f"Dense layer '{layer_name}' stream_len exceeds 32-bit range")
    if "weights" in checked and "weight_fp" in checked:
        raise ValueError(f"Dense layer '{layer_name}' cannot specify both weights and weight_fp")
    if "weight_fp" in checked:
        _format_unsigned_word(
            checked["weight_fp"], data_width, f"Dense layer '{layer_name}' weight_fp"
        )
    if "weights" in checked:
        raw_weights = checked["weights"]
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError(f"Dense layer '{layer_name}' weights must be a list or tuple")
        matrix = dense_weight_matrix(checked)
        if matrix is None:
            for idx, item in enumerate(raw_weights):
                _format_unsigned_word(
                    item, data_width, f"Dense layer '{layer_name}' weights[{idx}]"
                )
        else:
            _validate_weight_matrix(layer_name, checked, matrix, n_neurons, data_width)
    if dense_weight_matrix(checked) is None:
        for key in _SEED_KEYS:
            if key in checked:
                raise ValueError(
                    f"Dense layer '{layer_name}' {key} needs a 2-D weights matrix: "
                    "sc_dense_layer_core has fixed encoder seeds"
                )
    for key, default in (
        ("y_min_fp", 0),
        ("y_max_fp", 0x0200),
        ("cfg_leak", 0x0010),
        ("cfg_gain", 0x0100),
    ):
        _format_signed_word(
            checked.get(key, default), data_width, f"Dense layer '{layer_name}' {key}"
        )


def _validate_weight_matrix(
    layer_name: str,
    params: Mapping[str, object],
    matrix: list[list[object]],
    n_neurons: int,
    data_width: int,
) -> None:
    """Check a 2-D ``weights`` matrix, its seeds and its current range for ``sc_dense_matrix_layer``.

    Parameters
    ----------
    layer_name : str
        Sanitised Verilog layer identifier.
    params : Mapping[str, object]
        ``Dense`` layer parameters.
    matrix : list[list[object]]
        Weight rows, one per neuron.
    n_neurons : int
        Validated neuron count.
    data_width : int
        Word width of the weights and the current range.

    Raises
    ------
    ValueError
        If the row count or row widths are wrong, a weight or seed is out of range, or
        ``y_min_fp`` is not below ``y_max_fp``.
    """
    if len(matrix) != n_neurons:
        raise ValueError(
            f"Dense layer '{layer_name}' weights has {len(matrix)} rows; n_neurons is {n_neurons}"
        )
    width = len(matrix[0])
    if width == 0 or any(len(row) != width for row in matrix):
        raise ValueError(f"Dense layer '{layer_name}' weight rows must share one non-zero width")
    if "input_width" in params and params["input_width"] != width:
        raise ValueError(
            f"Dense layer '{layer_name}' weight rows have {width} entries; "
            f"input_width is {params['input_width']}"
        )
    for row_idx, row in enumerate(matrix):
        for col_idx, item in enumerate(row):
            _format_unsigned_word(
                item, data_width, f"Dense layer '{layer_name}' weights[{row_idx}][{col_idx}]"
            )
    for key in _SEED_KEYS:
        if key in params:
            seed = _require_non_bool_int(params[key], f"Dense layer '{layer_name}' {key}")
            if not 1 <= seed <= 0xFFFF:
                raise ValueError(f"Dense layer '{layer_name}' {key} must be in [1, 0xFFFF]")
    y_min = _require_non_bool_int(params.get("y_min_fp", 0), f"Dense layer '{layer_name}' y_min_fp")
    y_max = _require_non_bool_int(
        params.get("y_max_fp", 0x0200), f"Dense layer '{layer_name}' y_max_fp"
    )
    if y_min >= y_max:
        raise ValueError(
            f"Dense layer '{layer_name}' y_min_fp ({y_min}) must be below y_max_fp ({y_max})"
        )


def compute_dense_layer_widths(
    layers: list[dict[str, Any]],
    bus_width: int,
) -> list[tuple[int, int]]:
    """Compute ``(input_width, output_width)`` per ``Dense`` layer and reject mismatches.

    Parameters
    ----------
    layers : list[dict[str, Any]]
        Registered network layer entries.
    bus_width : int
        Default top-level input bus width.

    Returns
    -------
    list[tuple[int, int]]
        Ordered ``(input_width, output_width)`` pairs for ``Dense`` layers.
    """
    widths: list[tuple[int, int]] = []
    previous_width: int | None = None
    previous_name: str | None = None
    seen_dense: set[str] = set()
    for layer in layers:
        if layer["type"] != "Dense":
            continue
        name = str(layer["name"])
        if name in seen_dense:
            raise ValueError(f"duplicate layer name {name!r}")
        seen_dense.add(name)
        raw_params = _require_params_mapping(layer["params"], name)
        if "input_width" in raw_params:
            in_w = require_positive_int(raw_params["input_width"], "input_width")
        else:
            in_w = previous_width if previous_width is not None else bus_width
        if "output_width" in raw_params:
            out_w = require_positive_int(raw_params["output_width"], "output_width")
        else:
            out_w = require_positive_int(raw_params["n_neurons"], "n_neurons")
        if previous_width is not None and in_w != previous_width:
            raise ValueError(
                f"{previous_name} -> {name} width mismatch: "
                f"{previous_width} output bits cannot drive {in_w} input bits"
            )
        widths.append((in_w, out_w))
        previous_width = out_w
        previous_name = name
    return widths


def render_dense_layer_instance(
    *,
    comment_label: str,
    layer_name: str,
    params: Mapping[str, object],
    input_bus: str,
    output_bus: str,
    input_width: int,
) -> str:
    """Render one ``Dense`` layer instance bound to bitstream buses.

    A 2-D ``weights`` matrix renders ``sc_dense_matrix_layer``; any other form renders
    ``sc_dense_layer_core`` exactly as before.

    Parameters
    ----------
    comment_label : str
        Header comment line identifying the layer.
    layer_name : str
        Sanitised layer identifier.
    params : Mapping[str, object]
        Validated ``Dense`` layer parameters.
    input_bus : str
        Incoming 1-bit-per-channel Verilog signal name.
    output_bus : str
        Outgoing ``N_NEURONS``-bit spike Verilog signal name.
    input_width : int
        Bit width of ``input_bus`` (mapped to ``N_INPUTS``).

    Returns
    -------
    str
        Indented Verilog instantiation block.
    """
    matrix = dense_weight_matrix(params)
    if matrix is not None:
        return _render_matrix_layer_instance(
            comment_label=comment_label,
            layer_name=layer_name,
            params=params,
            matrix=matrix,
            input_bus=input_bus,
            output_bus=output_bus,
            input_width=input_width,
        )
    n_neurons = require_positive_int(params["n_neurons"], "n_neurons")
    dw = require_positive_int(params.get("data_width", 16), "data_width")
    stream_len = require_positive_int(params.get("stream_len", 256), "stream_len")
    if "weights" in params:
        raw_weights = params["weights"]
        if not isinstance(raw_weights, (list, tuple)):
            raise ValueError(f"Dense layer '{layer_name}' weights must be a list or tuple")
        if len(raw_weights) != input_width:
            raise ValueError(
                f"Dense layer '{layer_name}' weights length ({len(raw_weights)}) "
                f"must equal input_width ({input_width})"
            )
        words = [
            _format_unsigned_word(w, dw, f"Dense layer '{layer_name}' weights")
            for w in reversed(raw_weights)
        ]
        weight_expr = "{" + ", ".join(words) + "}"
    else:
        default_w = (1 << dw) - 1
        w_lit = _format_unsigned_word(params.get("weight_fp", default_w), dw, "weight_fp")
        weight_expr = f"{{{input_width}{{{w_lit}}}}}"

    y_min = _format_signed_word(params.get("y_min_fp", 0), dw, "y_min_fp")
    y_max = _format_signed_word(params.get("y_max_fp", 0x0200), dw, "y_max_fp")
    cfg_leak = _format_signed_word(params.get("cfg_leak", 0x0010), dw, "cfg_leak")
    cfg_gain = _format_signed_word(params.get("cfg_gain", 0x0100), dw, "cfg_gain")

    lines = [
        f"    // {comment_label}",
        f"    wire [{input_width * dw - 1}:0] {layer_name}_x_input_fp;",
        f"    wire [{dw - 1}:0] {layer_name}_i_t;",
        f"    wire {layer_name}_step_valid;",
        f"    wire {layer_name}_run_done;",
        f"    wire {layer_name}_running;",
        "    generate",
        f"        for (sc_in_idx = 0; sc_in_idx < {input_width}; sc_in_idx = sc_in_idx + 1) "
        f"begin : {layer_name}_pack_x",
        f"            assign {layer_name}_x_input_fp[sc_in_idx*{dw} +: {dw}] = "
        f"{{{dw}{{{input_bus}[sc_in_idx]}}}};",
        "        end",
        "    endgenerate",
        "    sc_dense_layer_core #(",
        f"        .N_INPUTS({input_width}),",
        f"        .N_NEURONS({n_neurons}),",
        f"        .DATA_WIDTH({dw})",
        f"    ) {layer_name}_inst (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .start_pulse(1'b1),",
        f"        .stream_len(32'd{stream_len}),",
        f"        .x_input_fp({layer_name}_x_input_fp),",
        f"        .weight_fp({weight_expr}),",
        f"        .y_min_fp({y_min}),",
        f"        .y_max_fp({y_max}),",
        f"        .cfg_leak({cfg_leak}),",
        f"        .cfg_gain({cfg_gain}),",
        f"        .I_t({layer_name}_i_t),",
        f"        .spikes({output_bus}),",
        f"        .step_valid({layer_name}_step_valid),",
        f"        .run_done({layer_name}_run_done),",
        f"        .running({layer_name}_running)",
        "    );",
        f"    wire _unused_{layer_name}_status = &{{1'b0, {layer_name}_i_t, "
        f"{layer_name}_step_valid, {layer_name}_run_done, {layer_name}_running, 1'b0}};",
        "",
    ]
    return "\n".join(lines)


def _render_matrix_layer_instance(
    *,
    comment_label: str,
    layer_name: str,
    params: Mapping[str, object],
    matrix: list[list[object]],
    input_bus: str,
    output_bus: str,
    input_width: int,
) -> str:
    """Render one ``sc_dense_matrix_layer`` instance with one weight row per neuron.

    Parameters
    ----------
    comment_label : str
        Header comment line identifying the layer.
    layer_name : str
        Sanitised layer identifier.
    params : Mapping[str, object]
        Validated ``Dense`` layer parameters.
    matrix : list[list[object]]
        Weight rows, one per neuron, each ``input_width`` entries long.
    input_bus : str
        Incoming 1-bit-per-channel Verilog signal name.
    output_bus : str
        Outgoing ``N_NEURONS``-bit spike Verilog signal name.
    input_width : int
        Bit width of ``input_bus`` (mapped to ``N_INPUTS``).

    Returns
    -------
    str
        Indented Verilog instantiation block.

    Raises
    ------
    ValueError
        If a weight row's width differs from ``input_width``.
    """
    n_neurons = require_positive_int(params["n_neurons"], "n_neurons")
    dw = require_positive_int(params.get("data_width", 16), "data_width")
    stream_len = require_positive_int(params.get("stream_len", 256), "stream_len")
    if any(len(row) != input_width for row in matrix):
        raise ValueError(
            f"Dense layer '{layer_name}' weight rows must have input_width ({input_width}) entries"
        )
    # weight_fp[(j * N_INPUTS + i) * DW +: DW] is row j, input i: the last word goes first.
    words = [
        _format_unsigned_word(item, dw, f"Dense layer '{layer_name}' weights")
        for row in reversed(matrix)
        for item in reversed(row)
    ]
    weight_expr = "{" + ", ".join(words) + "}"
    seeds = {
        key: _require_non_bool_int(params.get(key, default), key)
        for key, default in _DEFAULT_SEEDS.items()
    }
    y_min = _format_signed_word(params.get("y_min_fp", 0), dw, "y_min_fp")
    y_max = _format_signed_word(params.get("y_max_fp", 0x0200), dw, "y_max_fp")
    cfg_leak = _format_signed_word(params.get("cfg_leak", 0x0010), dw, "cfg_leak")
    cfg_gain = _format_signed_word(params.get("cfg_gain", 0x0100), dw, "cfg_gain")
    lines = [
        f"    // {comment_label}",
        f"    wire [{input_width * dw - 1}:0] {layer_name}_x_input_fp;",
        f"    wire {layer_name}_step_valid;",
        f"    wire {layer_name}_run_done;",
        f"    wire {layer_name}_running;",
        "    generate",
        f"        for (sc_in_idx = 0; sc_in_idx < {input_width}; sc_in_idx = sc_in_idx + 1) "
        f"begin : {layer_name}_pack_x",
        f"            assign {layer_name}_x_input_fp[sc_in_idx*{dw} +: {dw}] = "
        f"{{{dw}{{{input_bus}[sc_in_idx]}}}};",
        "        end",
        "    endgenerate",
        "    sc_dense_matrix_layer #(",
        f"        .N_INPUTS({input_width}),",
        f"        .N_NEURONS({n_neurons}),",
        f"        .DATA_WIDTH({dw}),",
        f"        .INPUT_SEED_BASE(16'h{seeds['input_seed_base']:04X}),",
        f"        .WEIGHT_SEED_BASE(16'h{seeds['weight_seed_base']:04X})",
        f"    ) {layer_name}_inst (",
        "        .clk(clk),",
        "        .rst_n(rst_n),",
        "        .start_pulse(1'b1),",
        f"        .stream_len(32'd{stream_len}),",
        f"        .x_input_fp({layer_name}_x_input_fp),",
        f"        .weight_fp({weight_expr}),",
        f"        .y_min_fp({y_min}),",
        f"        .y_max_fp({y_max}),",
        f"        .cfg_leak({cfg_leak}),",
        f"        .cfg_gain({cfg_gain}),",
        f"        .spikes({output_bus}),",
        f"        .step_valid({layer_name}_step_valid),",
        f"        .run_done({layer_name}_run_done),",
        f"        .running({layer_name}_running)",
        "    );",
        f"    wire _unused_{layer_name}_status = &{{1'b0, {layer_name}_step_valid, "
        f"{layer_name}_run_done, {layer_name}_running, 1'b0}};",
        "",
    ]
    return "\n".join(lines)


class AEREmitter:
    """Emit a research-stage AER wrapper around the clocked HDL dense path.

    Each ``Dense`` layer of the compute pipeline binds ``sc_dense_layer_core`` for a 1-D
    ``weights`` or one ``weight_fp`` (one shared current, so one signal; seed parameters
    refused) or ``sc_dense_matrix_layer`` for a 2-D ``weights`` matrix (per-row weights,
    seeds and current range). The final spike vector is wrapped in a 4-phase AER-style
    request/acknowledge interface. It is not a quasi-delay-insensitive asynchronous network
    replacement.
    """

    def __init__(self, module_name: str = "sc_network_async_aer", bus_width: int = 8) -> None:
        """Initialise an AER wrapper emitter.

        Parameters
        ----------
        module_name : str
            Sanitised Verilog module identifier.
        bus_width : int
            Default top-level input bus width in bits.
        """
        self.module_name = sanitize_ident(module_name, context="module name")
        self.bus_width = require_positive_int(bus_width, "bus_width")
        self.layers: list[dict[str, Any]] = []

    def add_layer(self, layer_type: str, name: str, params: dict[str, Any]) -> None:
        """Append one ``Dense`` layer specification to the AER wrapper.

        Parameters
        ----------
        layer_type : str
            Layer kind (must be ``"Dense"``).
        name : str
            Layer instance prefix.
        params : dict[str, object]
            Layer parameter mapping.
        """
        self.layers.append(
            {
                "type": layer_type,
                "name": sanitize_ident(name, context="layer name"),
                "params": params,
            }
        )

    def generate(self) -> str:
        """Emit complete Verilog source for the clocked AER wrapper.

        Returns
        -------
        str
            Generated Verilog module text.
        """
        self._validate_layers()
        layer_widths = compute_dense_layer_widths(self.layers, self.bus_width)
        input_width = layer_widths[0][0] if layer_widths else self.bus_width
        spike_width = layer_widths[-1][1] if layer_widths else self.bus_width
        addr_width = max(1, ceil(log2(spike_width)))

        lines = [
            f"module {self.module_name} (",
            "    input wire clk,",
            "    input wire rst_n,",
            f"    input wire [{input_width - 1}:0] input_bus,",
            "    input wire aer_ack,",
            "    output reg aer_req,",
            f"    output reg [{addr_width - 1}:0] aer_addr,",
            f"    output wire [{spike_width - 1}:0] output_bus",
            ");",
            "",
            "    // Research boundary: sync compute path with AER output wrapper.",
            "    // This is not a full asynchronous micropipeline implementation.",
            f"    wire [{spike_width - 1}:0] spike_vector;",
        ]
        for i in range(len(layer_widths) - 1):
            lines.append(f"    wire [{layer_widths[i][1] - 1}:0] layer_{i}_to_{i + 1};")
        if layer_widths:
            lines.append("    genvar sc_in_idx;")
        lines.append("")

        for i, layer in enumerate(self.layers):
            out_sig = "spike_vector" if i == len(layer_widths) - 1 else f"layer_{i}_to_{i + 1}"
            in_sig = "input_bus" if i == 0 else f"layer_{i - 1}_to_{i}"
            raw_params = _require_params_mapping(layer["params"], str(layer["name"]))
            lines.append(
                render_dense_layer_instance(
                    comment_label=f"Sync layer {i}: {layer['name']}",
                    layer_name=str(layer["name"]),
                    params=raw_params,
                    input_bus=in_sig,
                    output_bus=out_sig,
                    input_width=layer_widths[i][0],
                )
            )

        if not self.layers:
            lines.extend(
                [
                    f"    assign spike_vector = {spike_width}'b0;",
                    "    wire _unused_empty_inputs = &{1'b0, input_bus, 1'b0};",
                    "",
                ]
            )

        lines.extend(
            [
                "    assign output_bus = spike_vector;",
                "    wire spike_valid = |spike_vector;",
                "",
                f"    function [{addr_width - 1}:0] first_hot_index;",
                f"        input [{spike_width - 1}:0] vector;",
                "        integer k;",
                "        reg found;",
                "        begin",
                f"            first_hot_index = {addr_width}'d0;",
                "            found = 1'b0;",
                f"            for (k = 0; k < {spike_width}; k = k + 1) begin",
                "                if (!found && vector[k]) begin",
                f"                    first_hot_index = k[{addr_width - 1}:0];",
                "                    found = 1'b1;",
                "                end",
                "            end",
                "        end",
                "    endfunction",
                "",
                f"    wire [{addr_width - 1}:0] encoded_addr = first_hot_index(spike_vector);",
                f"    reg [{spike_width - 1}:0] event_vector;",
                f"    reg [{spike_width - 1}:0] acknowledged_vector;",
                "    wire new_spike_vector = spike_valid && (spike_vector != acknowledged_vector);",
                "",
                "    always @(posedge clk or negedge rst_n) begin",
                "        if (!rst_n) begin",
                "            aer_req <= 1'b0;",
                f"            aer_addr <= {addr_width}'d0;",
                f"            event_vector <= {spike_width}'d0;",
                f"            acknowledged_vector <= {spike_width}'d0;",
                "        end else begin",
                "            if (!spike_valid) begin",
                f"                acknowledged_vector <= {spike_width}'d0;",
                "            end",
                "            if (!aer_req && new_spike_vector) begin",
                "                aer_req <= 1'b1;",
                "                aer_addr <= encoded_addr;",
                "                event_vector <= spike_vector;",
                "            end else if (aer_req && aer_ack) begin",
                "                aer_req <= 1'b0;",
                "                acknowledged_vector <= event_vector;",
                "            end",
                "        end",
                "    end",
                "",
                "endmodule",
                "",
            ]
        )
        return "\n".join(lines)

    def _validate_layers(self) -> None:
        for layer in self.layers:
            if layer["type"] != "Dense":
                raise ValueError(
                    f"unsupported async AER layer type '{layer['type']}' "
                    f"for layer '{layer['name']}'"
                )
            validate_dense_layer_params(str(layer["name"]), layer["params"])
