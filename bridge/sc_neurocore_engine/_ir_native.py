# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Typed native compute-graph contracts

"""Static contracts for the compute-graph subset of the native PyO3 module.

These protocols describe the real functions in engine/src/ir/bindings.rs.
The module cast is the single extension boundary; no dynamic attribute fallback
is provided for missing or misspelled members.
"""

from __future__ import annotations

from importlib import import_module
from typing import Callable, NotRequired, Protocol, TypedDict, cast


class DenseFoldPlan(TypedDict):
    """Time-multiplexing plan of a dense layer over a target's DSP budget."""

    n_inputs: int
    n_outputs: int
    mac_count: int
    dsp_budget: int
    output_parallelism: int
    input_parallelism: int
    dsp_per_cycle: int
    input_fold_factor: int
    output_fold_factor: int
    compute_cycles: int
    fold_required: bool
    fits_dsp_budget: bool


class HeuristicResourceReport(TypedDict):
    """Uncalibrated heuristic estimate of SystemVerilog for a Zynq device.

    ``calibration`` is ``"uncalibrated heuristic"``: no number here has been
    measured against synthesis. Budgets equal to ``2**32 - 1`` mean the target
    names no limit for that resource.
    """

    target_name: str
    device_part: str
    clock_mhz: int
    calibration: str
    lut_estimated: int
    ff_estimated: int
    dsp_estimated: int
    bram_36k_estimated: int
    uram_estimated: int
    critical_path_estimate_ns: float
    dsp_budget: int
    bram_36k_budget: int
    uram_budget: int
    fits_dsp_budget: bool
    fits_bram_budget: bool
    fits_uram_budget: bool
    dense_fold_plan: DenseFoldPlan | None


class Ice40ResourceReport(TypedDict):
    """Report of SystemVerilog emitted for the iCE40 UltraPlus UP5K.

    When ``static_estimate`` is true the graph is one calibrated block inside
    its measured domain, and ``logic_cells_estimated`` and
    ``critical_path_estimate_ns`` are upper bounds measured with the toolchain
    named in ``calibration``. When it is false the report carries the reason
    and none of the estimate keys: synthesis with place and route decides.
    """

    target_name: str
    device_part: str
    clock_mhz: int
    clock_period_ns: float
    calibration: str
    logic_cell_budget: int
    ram_block_budget: int
    dsp_budget: int
    static_estimate: bool
    static_estimate_reason: str | None
    block: NotRequired[str]
    logic_cells_estimated: NotRequired[int]
    critical_path_estimate_ns: NotRequired[float]
    fits_logic_cells: NotRequired[bool]
    meets_timing_estimate: NotRequired[bool]


class NativeGraph(Protocol):
    """Native graph properties and operation counts."""

    @property
    def name(self) -> str:
        """Return the graph's module name."""
        ...

    @property
    def source_payload(self) -> str | None:
        """Return exact UTF-8 provenance when bound."""
        ...

    @property
    def source_sha256(self) -> str | None:
        """Return the provenance checksum, without an authentication claim."""
        ...

    @property
    def source_graph_sha256(self) -> str | None:
        """Return the bound canonical graph checksum."""
        ...

    def len(self) -> int:
        """Count operations including ports and constants."""
        ...

    def num_inputs(self) -> int:
        """Count declared module inputs."""
        ...

    def num_outputs(self) -> int:
        """Count declared module outputs."""
        ...


class NativeBuilder(Protocol):
    """Actual positional PyO3 graph-builder signatures."""

    def bind_source(self, payload: str) -> None:
        """Bind nonempty source bytes to the current complete graph, once."""
        ...

    def runtime_check(
        self, condition: int, kind: str = "overflow", enable: int | None = None
    ) -> int:
        """Append a non-value Boolean simulation check, excluded from synthesis."""
        ...

    def state(self, initial: int, ty: str) -> int:
        """Declare a scalar register with an earlier constant reset value."""
        ...

    def store(self, state: int, next: int, enable: int | None = None) -> int:
        """Bind one atomic update, optionally enabled by a Boolean value."""
        ...

    def word_delay(self, input: int, ty: str) -> int:
        """Capture a full integer word with one clock latency and asynchronous zero reset.

        Parameters
        ----------
        input : int
            Native SSA identifier of the word sampled at each rising clock.
        ty : str
            Identical input/output iN or uN geometry, including widths above 64 bits.

        Returns
        -------
        int
            Native SSA identifier of the registered physical word.
        """
        ...

    def word_binary(self, kind: str, lhs: int, rhs: int, ty: str) -> int:
        """Append modulo-width integer arithmetic."""
        ...

    def word_unary(self, kind: str, input: int, ty: str) -> int:
        """Append integer negation or complement."""
        ...

    def word_compare(self, predicate: str, lhs: int, rhs: int) -> int:
        """Append an integer comparison returning Bool."""
        ...

    def word_resize(self, input: int, ty: str) -> int:
        """Resize an integer word explicitly."""
        ...

    def word_bitcast(self, input: int, ty: str) -> int:
        """Reinterpret a scalar word without changing its bits."""
        ...

    def word_select(self, condition: int, when_true: int, when_false: int, ty: str) -> int:
        """Select a scalar word using a Boolean condition."""
        ...

    def word_round(self, input: int, shift: int, mode: str, ty: str) -> int:
        """Divide an integer word by a power of two."""
        ...

    def word_clamp(self, input: int, min: int, max: int, ty: str) -> int:
        """Saturate an integer word at explicit inclusive bounds."""
        ...

    def input(self, name: str, ty: str) -> int:
        """Declare an input using the native type parser."""
        ...

    def output(self, name: str, source_id: int) -> int:
        """Forward an SSA value to an output port."""
        ...

    def constant_f64(self, value: float, ty: str) -> int:
        """Quantize a floating scalar using its declared type."""
        ...

    def constant_i64(self, value: int, ty: str) -> int:
        """Store an already encoded integer word."""
        ...

    def constant_f64_vec(self, values: list[float], ty: str) -> int:
        """Store a typed vector, inferring shape from an element type."""
        ...

    def kuramoto_step(self, phases_id: int, omega_id: int, coupling_id: int, dt: float) -> int:
        """Append a coupled oscillator step."""
        ...

    def graph_forward(
        self, features_id: int, adjacency_id: int, n_nodes: int, n_features: int
    ) -> int:
        """Append degree-normalized graph aggregation."""
        ...

    def softmax_attention(self, q_id: int, k_id: int, v_id: int, dim_k: int) -> int:
        """Append single-head attention."""
        ...

    def encode(self, prob_id: int, length: int, seed: int) -> int:
        """Encode a typed probability into a streamed bit."""
        ...

    def bitwise_and(self, lhs_id: int, rhs_id: int) -> int:
        """Append stochastic multiplication."""
        ...

    def popcount(self, input_id: int) -> int:
        """Append Hamming weight extraction."""
        ...

    def decode(self, input_id: int) -> int:
        """Decode binary observations to a Q8.8 Rate result."""
        ...

    def lif_step(
        self,
        current_id: int,
        leak_id: int,
        gain_id: int,
        noise_id: int,
        data_width: int,
        fraction: int,
        v_rest: int,
        v_reset: int,
        v_threshold: int,
        refractory_period: int,
    ) -> int:
        """Append a configured fixed-point LIF step."""
        ...

    def dense_forward(
        self,
        inputs_id: int,
        weights_id: int,
        leak_id: int,
        gain_id: int,
        n_inputs: int,
        n_neurons: int,
        data_width: int,
        stream_length: int,
        seed_base: int,
        y_min: int,
        y_max: int,
    ) -> int:
        """Append a seeded row-major dense matrix layer."""
        ...

    def scale(self, input_id: int, factor: float) -> int:
        """Append multiplication by a real constant."""
        ...

    def offset(self, input_id: int, offset_val: float) -> int:
        """Append addition of a real constant."""
        ...

    def div_const(self, input_id: int, divisor: int) -> int:
        """Append unsigned packed-word division with a low-64-bit quotient."""
        ...

    def build(self) -> NativeGraph:
        """Consume the builder exactly once."""
        ...


class NativeIR(Protocol):
    """Native module members used by the Python IR facade."""

    ScGraphBuilder: Callable[[str], NativeBuilder]

    def ir_verify(self, graph: NativeGraph) -> list[str] | None:
        """Return native verification diagnostics or None."""
        ...

    def ir_print(self, graph: NativeGraph) -> str:
        """Serialize all graph operations and parameters."""
        ...

    def ir_parse(self, text: str) -> NativeGraph:
        """Parse persisted graph text or raise ValueError."""
        ...

    def ir_emit_sv(self, graph: NativeGraph) -> str:
        """Emit synthesizable SystemVerilog or raise ValueError."""
        ...

    def ir_emit_sv_with_target(
        self, graph: NativeGraph, target: str, clock_mhz: int
    ) -> tuple[str, Ice40ResourceReport | HeuristicResourceReport]:
        """Emit SystemVerilog for a device target with that target's report."""
        ...

    def ir_emit_mlir(self, graph: NativeGraph) -> str:
        """Emit native CIRCT MLIR or raise ValueError."""
        ...


native = cast(NativeIR, import_module("sc_neurocore_engine.sc_neurocore_engine"))
