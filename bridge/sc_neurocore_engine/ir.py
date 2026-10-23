# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — IR - Python API for compute graph construction and

"""SC-NeuroCore IR - Python API for compute graph construction and compilation."""

from __future__ import annotations

from sc_neurocore_engine._ir_native import NativeGraph as _ScGraph, native as _native


class ScGraphBuilder:
    """Fluent builder for SC compute graphs.

    Example::

        b = ScGraphBuilder("my_synapse")
        x = b.input("x_prob", "rate")
        w = b.input("w_prob", "rate")
        x_enc = b.encode(x, length=1024, seed=0xACE1)
        w_enc = b.encode(w, length=1024, seed=0xBEEF)
        product = b.bitwise_and(x_enc, w_enc)
        rate = b.decode(product)
        b.output("firing_rate", rate)
        graph = b.build()
    """

    def __init__(self, name: str) -> None:
        self._builder = _native.ScGraphBuilder(name)

    def bind_source(self, payload: str) -> None:
        """Bind exact UTF-8 provenance after all graph operations are appended.

        Parameters
        ----------
        payload : str
            Nonempty source description. The equation compiler supplies its
            canonical model/numeric JSON contract, rather than executable RTL.

        Raises
        ------
        ValueError
            If the source is empty, already bound or the builder was consumed.

        Notes
        -----
        Later operations invalidate verification and emission; identity is never
        silently rebound. Checksums detect damage or stale bindings and do not
        authenticate the author or prove the description's semantic accuracy.
        """
        self._builder.bind_source(payload)

    def runtime_check(
        self, condition: int, kind: str = "overflow", enable: int | None = None
    ) -> int:
        """Append an effectful simulation invariant without producing a data word.

        Parameters
        ----------
        condition : int
            Earlier Bool SSA identifier; false fails when enabled.
        kind : str, optional
            Closed diagnostic code, "overflow" or "domain".
        enable : int or None, optional
            Earlier Bool control. None checks unconditionally, including reset.

        Returns
        -------
        int
            Terminator identifier for provenance; cannot be consumed as a value.

        Raises
        ------
        ValueError
            The kind is unknown or the builder has already been consumed.
        OverflowError
            An identifier is outside u32 or the SSA allocator is exhausted.

        Notes
        -----
        The native verifier checks operand definitions and Boolean types.
        Simulation checks combinationally, including time zero; a false enable
        suppresses the effect. Defining SYNTHESIS excludes checks without
        changing data words or registers. There is no implicit reset gate.
        """
        return self._builder.runtime_check(condition, kind, enable)

    def state(self, initial: int, ty: str) -> int:
        """Declare current register storage with an asynchronous constant reset.

        Parameters
        ----------
        initial : int
            SSA identifier of an earlier scalar constant with exactly type ty.
        ty : str
            Bool, rate, fixed-point, signed or unsigned scalar of 1 to 64 bits.

        Returns
        -------
        int
            Identifier of the current register value.

        Notes
        -----
        Verification requires exactly one subsequent store. All stores commit
        simultaneously at the rising clock edge. Active-low rst_n restores the
        quantised constant asynchronously, regardless of the store enable.
        """
        return self._builder.state(initial, ty)

    def store(self, state: int, next: int, enable: int | None = None) -> int:
        """Bind one atomic update to a declared state register.

        Parameters
        ----------
        state : int
            SSA identifier returned by state().
        next : int
            SSA value with exactly the register type, including binary point.
        enable : int or None, optional
            SSA Bool value; false holds the current state. None always updates.

        Returns
        -------
        int
            Terminator identifier, which cannot be consumed by another operation.

        Notes
        -----
        Missing or duplicate drivers, wrong types, non-State targets and
        terminator operands are rejected by graph verification and emission.
        """
        return self._builder.store(state, next, enable)

    def word_delay(self, input: int, ty: str) -> int:
        """Capture an integer word for one clock with asynchronous zero reset.

        Parameters
        ----------
        input : int
            Earlier native SSA identifier sampled on every rising edge of clk.
        ty : str
            Explicit iN or uN type, identical to the input. Register widths
            above 64 bits preserve full intermediate products.

        Returns
        -------
        int
            Identifier of the registered word, delayed by exactly one clock.

        Notes
        -----
        Active-low rst_n clears every bit asynchronously, regardless of clk.
        Consecutive calls form separate one-clock stages. There is no enable,
        rescaling, saturation or implicit truncation. Graph verification and
        both emitters reject missing producers and incompatible word types.
        """
        return self._builder.word_delay(input, ty)

    def word_binary(self, kind: str, lhs: int, rhs: int, ty: str) -> int:
        """Append modulo-width integer arithmetic.

        Parameters
        ----------
        kind : str
            One of add, sub, mul, div, rem, and, or, xor, shl, lshr, ashr.
        lhs, rhs : int
            SSA operands with the same declared integer type.
        ty : str
            Explicit signed iN or unsigned uN result type.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Division truncates toward zero and returns zero for a zero divisor.
        MIN / -1 wraps. Shift counts use unsigned physical bits; shifts beyond
        the width return zero, or the sign fill for ashr.
        """
        return self._builder.word_binary(kind, lhs, rhs, ty)

    def word_unary(self, kind: str, input: int, ty: str) -> int:
        """Append integer negation or complement.

        Parameters
        ----------
        kind : str
            neg or not.
        input : int
            SSA integer operand.
        ty : str
            Explicit result type, identical to the operand.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Negation wraps at the declared width.
        """
        return self._builder.word_unary(kind, input, ty)

    def word_compare(self, predicate: str, lhs: int, rhs: int) -> int:
        """Append an integer comparison returning Bool.

        Parameters
        ----------
        predicate : str
            eq, ne, lt, le, gt or ge.
        lhs, rhs : int
            SSA operands with identical integer types.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Ordering follows operand signedness. The result is one Boolean bit.
        """
        return self._builder.word_compare(predicate, lhs, rhs)

    def word_resize(self, input: int, ty: str) -> int:
        """Resize an integer word explicitly.

        Parameters
        ----------
        input : int
            SSA integer or Boolean operand.
        ty : str
            Explicit iN or uN destination type.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Widening sign-extends signed inputs and zero-extends unsigned or Bool
        inputs. Narrowing discards high bits. No saturation or rescaling occurs.
        """
        return self._builder.word_resize(input, ty)

    def word_bitcast(self, input: int, ty: str) -> int:
        """Reinterpret a scalar word without changing its bits.

        Parameters
        ----------
        input : int
            SSA scalar operand.
        ty : str
            Scalar destination type with the same physical width.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Supports fixed-point, rate, integer and Bool scalars. This operation
        does not move a binary point or quantize a value.
        """
        return self._builder.word_bitcast(input, ty)

    def word_select(self, condition: int, when_true: int, when_false: int, ty: str) -> int:
        """Select a scalar word using a Boolean condition.

        Parameters
        ----------
        condition : int
            SSA Bool operand.
        when_true, when_false : int
            SSA branch operands with identical scalar types.
        ty : str
            Explicit branch and result type.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Both branches are hardware values, not short-circuit expressions.
        """
        return self._builder.word_select(condition, when_true, when_false, ty)

    def word_round(self, input: int, shift: int, mode: str, ty: str) -> int:
        """Divide an integer word by a power of two.

        Parameters
        ----------
        input : int
            SSA integer operand.
        shift : int
            Nonnegative exponent smaller than the word width.
        mode : str
            floor, half_up, toward_zero, half_even or half_away.
        ty : str
            Explicit type, identical to the operand.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        half_up rounds ties toward positive infinity; half_away rounds ties
        away from zero. Rounding introduces no intermediate overflow.
        """
        return self._builder.word_round(input, shift, mode, ty)

    def word_clamp(self, input: int, min: int, max: int, ty: str) -> int:
        """Saturate an integer word at explicit inclusive bounds.

        Parameters
        ----------
        input : int
            SSA integer operand.
        min, max : int
            Ordered signed-64-bit literals fitting the integer type.
        ty : str
            Explicit type, identical to the operand.

        Returns
        -------
        int
            Identifier of the appended graph value.

        Notes
        -----
        Apply only where the source commits saturation; arithmetic itself wraps.
        """
        return self._builder.word_clamp(input, min, max, ty)

    def input(self, name: str, ty: str = "rate") -> int:
        """Add a typed input port. Returns value ID."""
        return self._builder.input(name, ty)

    def output(self, name: str, source_id: int) -> int:
        """Expose an existing graph value as a named output port.

        Parameters
        ----------
        name : str
            Name of the emitted hardware output port.
        source_id : int
            Identifier of the graph value connected to the port.

        Returns
        -------
        int
            Identifier of the appended output operation.
        """
        return self._builder.output(name, source_id)

    def constant_f64(self, value: float, ty: str = "rate") -> int:
        """Add a float constant."""
        return self._builder.constant_f64(value, ty)

    def constant_i64(self, value: int, ty: str = "i32") -> int:
        """Add an integer constant."""
        return self._builder.constant_i64(value, ty)

    def constant_f64_vec(self, values: list[float], ty: str = "rate") -> int:
        """Add a float-vector constant."""
        return self._builder.constant_f64_vec(values, ty)

    def kuramoto_step(
        self,
        phases_id: int,
        omega_id: int,
        coupling_id: int,
        dt: float = 0.01,
    ) -> int:
        """Add a single Kuramoto integration step over an N×N coupling matrix.

        ``phases_id`` and ``omega_id`` are length-``N`` vector constants and
        ``coupling_id`` is the row-major ``N×N`` matrix ``K_nm``.
        """
        return self._builder.kuramoto_step(phases_id, omega_id, coupling_id, dt)

    def graph_forward(
        self,
        features_id: int,
        adjacency_id: int,
        n_nodes: int,
        n_features: int,
    ) -> int:
        """Add a degree-normalised graph aggregation over an adjacency matrix.

        ``features_id`` is the ``n_nodes × n_features`` (node-major) vector constant
        and ``adjacency_id`` the row-major ``n_nodes × n_nodes`` matrix. The result is
        ``agg[i][f] = (Σ_j adj[i][j]·feat[j][f]) / degree[i]``.
        """
        return self._builder.graph_forward(features_id, adjacency_id, n_nodes, n_features)

    def softmax_attention(self, q_id: int, k_id: int, v_id: int, dim_k: int) -> int:
        """Add single-head scaled-dot-product softmax attention.

        ``q_id`` is ``q_rows × dim_k``, ``k_id`` is ``k_rows × dim_k`` and ``v_id`` is
        ``k_rows × v_cols`` (row-major vector constants). The result is
        ``softmax(Q·Kᵀ / sqrt(dim_k)) · V`` with shapes inferred from the operand lengths.
        """
        return self._builder.softmax_attention(q_id, k_id, v_id, dim_k)

    def encode(self, prob_id: int, length: int = 1024, seed: int = 0xACE1) -> int:
        """Add Bernoulli bitstream encoding."""
        return self._builder.encode(prob_id, length, seed)

    def bitwise_and(self, lhs_id: int, rhs_id: int) -> int:
        """Add bitwise AND (SC multiply)."""
        return self._builder.bitwise_and(lhs_id, rhs_id)

    def popcount(self, input_id: int) -> int:
        """Append a u64 packed population count or a held temporal window count.

        Temporal windows start after reset and sample the bits before each
        rising clock edge. The returned integer identifies the graph value.
        """
        return self._builder.popcount(input_id)

    def decode(self, input_id: int) -> int:
        """Append an explicit Q8.8 probability decoder.

        The circuit floors ``count * 256 / observations``. A temporal stream
        holds its completed-window result until the next window completes.
        The returned integer identifies the graph value, not its measured rate.
        """
        return self._builder.decode(input_id)

    def lif_step(
        self,
        current_id: int,
        leak_id: int,
        gain_id: int,
        noise_id: int,
        *,
        data_width: int = 16,
        fraction: int = 8,
        v_rest: int = 0,
        v_reset: int = 0,
        v_threshold: int = 256,
        refractory_period: int = 2,
    ) -> int:
        """Add a LIF neuron step."""
        return self._builder.lif_step(
            current_id,
            leak_id,
            gain_id,
            noise_id,
            data_width,
            fraction,
            v_rest,
            v_reset,
            v_threshold,
            refractory_period,
        )

    def dense_forward(
        self,
        inputs_id: int,
        weights_id: int,
        leak_id: int,
        gain_id: int,
        *,
        n_inputs: int = 3,
        n_neurons: int = 7,
        data_width: int = 16,
        stream_length: int = 1024,
        seed_base: int = 0xACE1,
        y_min: int = 0,
        y_max: int = 256,
    ) -> int:
        """Add a row-major dense layer with signed Q8 state and current words.

        Inputs and weights are vectors of Rate or signed FixedPoint probabilities.
        Weights contain ``n_neurons * n_inputs`` words. Their binary points may
        differ from the state format. ``seed_base`` sets the input seeds; the
        weight seed base is its 16-bit successor. Current endpoints must fit
        ``data_width``. The default upper endpoint is 256 (one in Q8).
        """
        return self._builder.dense_forward(
            inputs_id,
            weights_id,
            leak_id,
            gain_id,
            n_inputs,
            n_neurons,
            data_width,
            stream_length,
            seed_base,
            y_min,
            y_max,
        )

    def scale(self, input_id: int, factor: float) -> int:
        """Scale a value by a constant factor."""
        return self._builder.scale(input_id, factor)

    def offset(self, input_id: int, offset_val: float) -> int:
        """Add a constant offset."""
        return self._builder.offset(input_id, offset_val)

    def div_const(self, input_id: int, divisor: int) -> int:
        """Divide a packed integer word by a positive unsigned constant.

        The result retains the low 64 quotient bits after full-width division.
        Signed integer storage is interpreted as an unsigned bit pattern.
        Temporal streams require population counting or decoding first.
        """
        return self._builder.div_const(input_id, divisor)

    def build(self) -> ScGraph:
        """Consume the builder and return a ScGraph for explicit verification."""
        raw = self._builder.build()
        return ScGraph(raw)


class ScGraph:
    """An SC compute graph (DAG of SC operations).

    Typically constructed via ``ScGraphBuilder.build()``,
    or parsed from text format via ``parse_ir()``.
    """

    def __init__(self, _inner: _ScGraph) -> None:
        self._inner = _inner

    @property
    def name(self) -> str:
        """Return the graph name used for the emitted hardware module."""
        return self._inner.name

    @property
    def source_payload(self) -> str | None:
        """Return exact bound UTF-8 provenance, or None for legacy graphs."""
        return self._inner.source_payload

    @property
    def source_sha256(self) -> str | None:
        """Return the exact source checksum, without an authentication claim."""
        return self._inner.source_sha256

    @property
    def source_graph_sha256(self) -> str | None:
        """Return the checksum of canonical graph text without source metadata."""
        return self._inner.source_graph_sha256

    def __len__(self) -> int:
        """Count all graph operations, including constants and ports."""
        return self._inner.len()

    def __repr__(self) -> str:
        """Show the native graph name and operation count."""
        return repr(self._inner)

    @property
    def num_inputs(self) -> int:
        """Number of input ports."""
        return self._inner.num_inputs()

    @property
    def num_outputs(self) -> int:
        """Number of output ports."""
        return self._inner.num_outputs()

    def verify(self) -> list[str] | None:
        """Verify the graph. Returns None if valid, or list of error strings."""
        return _native.ir_verify(self._inner)

    def to_text(self) -> str:
        """Serialize to stable text format."""
        return _native.ir_print(self._inner)

    def emit_sv(self) -> str:
        """Emit SystemVerilog from this graph."""
        return _native.ir_emit_sv(self._inner)

    def emit_mlir(self) -> str:
        """Emit CIRCT HW/Comb/Seq text through the native compiler entry point."""
        return _native.ir_emit_mlir(self._inner)


def parse_ir(text: str) -> ScGraph:
    """Parse an SC graph from its text format."""
    raw = _native.ir_parse(text)
    return ScGraph(raw)
