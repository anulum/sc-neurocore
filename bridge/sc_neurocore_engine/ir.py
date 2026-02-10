"""SC-NeuroCore IR - Python API for compute graph construction and compilation."""

from __future__ import annotations

from sc_neurocore_engine.sc_neurocore_engine import (
    ScGraph as _ScGraph,
    ScGraphBuilder as _ScGraphBuilder,
    ir_verify as _verify,
    ir_print as _print,
    ir_parse as _parse,
    ir_emit_sv as _emit_sv,
)


class ScGraphBuilder:
    """Fluent builder for SC compute graphs.

    Example::

        b = ScGraphBuilder("my_synapse")
        x = b.input("x_prob", "rate")
        w = b.input("w_prob", "rate")
        x_enc = b.encode(x, length=1024, seed=0xACE1)
        w_enc = b.encode(w, length=1024, seed=0xBEEF)
        product = b.bitwise_and(x_enc, w_enc)
        count = b.popcount(product)
        rate = b.div_const(count, 1024)
        b.output("firing_rate", rate)
        graph = b.build()
    """

    def __init__(self, name: str) -> None:
        self._builder = _ScGraphBuilder(name)

    def input(self, name: str, ty: str = "rate") -> int:
        """Add a typed input port. Returns value ID."""
        return self._builder.input(name, ty)

    def output(self, name: str, source_id: int) -> int:
        """Add an output port."""
        return self._builder.output(name, source_id)

    def constant_f64(self, value: float, ty: str = "rate") -> int:
        """Add a float constant."""
        return self._builder.constant_f64(value, ty)

    def constant_i64(self, value: int, ty: str = "i32") -> int:
        """Add an integer constant."""
        return self._builder.constant_i64(value, ty)

    def encode(self, prob_id: int, length: int = 1024, seed: int = 0xACE1) -> int:
        """Add Bernoulli bitstream encoding."""
        return self._builder.encode(prob_id, length, seed)

    def bitwise_and(self, lhs_id: int, rhs_id: int) -> int:
        """Add bitwise AND (SC multiply)."""
        return self._builder.bitwise_and(lhs_id, rhs_id)

    def popcount(self, input_id: int) -> int:
        """Add Hamming weight extraction."""
        return self._builder.popcount(input_id)

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
        y_max: int = 65535,
    ) -> int:
        """Add a dense layer forward pass."""
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
        """Divide by a constant."""
        return self._builder.div_const(input_id, divisor)

    def build(self) -> ScGraph:
        """Consume the builder and return a verified ScGraph."""
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
        """Graph name."""
        return self._inner.name

    def __len__(self) -> int:
        return self._inner.len()

    def __repr__(self) -> str:
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
        return _verify(self._inner)

    def to_text(self) -> str:
        """Serialize to stable text format."""
        return _print(self._inner)

    def emit_sv(self) -> str:
        """Emit SystemVerilog from this graph."""
        return _emit_sv(self._inner)



def parse_ir(text: str) -> ScGraph:
    """Parse an SC graph from its text format."""
    raw = _parse(text)
    return ScGraph(raw)
