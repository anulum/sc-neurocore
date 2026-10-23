<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Compute graph text and hardware parameter preservation
-->

# Compute graph text

The Rust `ir::printer::print` and `ir::parser::parse` functions serialize
the native compute graph. The Python engine bridge exposes these as
`ScGraph.to_text()` and `sc_neurocore_engine.ir.parse_ir(text)`.

For compound neuron operations, the text retains every hardware parameter:

| Operation | Parameter attributes |
|---|---|
| `sc.lif_step` | `dw`, `frac`, `vr` (rest voltage), `reset`, `vt`, `rp` |
| `sc.dense_forward` | `ni`, `nn`, `len`, `dw`, `input_seed`, `weight_seed`, `y_min`, `y_max` |
| `sc.dcls_layer` | `taps`, `depth`, `dw`, `frac`, `ptr`, `offsets` |

The operand attributes (`leak`, `gain`, `noise`, `weights`, `centre`,
`sigma`) remain SSA value references. Voltage and current-mapping values
are signed integer words; seeds are unsigned decimal integers. Ordered DCLS
offsets use semicolons, for example `offsets=[7;0;7]`. Order and repeated
offsets are preserved. Saving and loading a graph retains the emitted
SystemVerilog and its target resource report for the same target.

Older LIF text omitted rest and reset voltages; older dense text omitted
word width, encoder seeds and current mapping. Omitted attributes retain
their historical defaults. Values that an old producer omitted cannot be
recovered from that text: regenerate it from the original configured graph.
Explicit malformed, overflowing, duplicate or unknown attributes are errors,
and required operands cannot be omitted.

Older DCLS text omitted pointer width and tap offsets and was not accepted
by the parser. The parser now supports DCLS, requires its offset list, and
uses the historical `DclsParams` defaults for any omitted scalar parameter.
Regenerate old DCLS text from the original graph to retain its configuration.

These operations describe hardware compute primitives. This serialization
contract does not establish arbitrary equation lowering or physical hardware
validation. Graph verification, numerical equivalence and board measurements
are separate checks.

For source-backed scalar/state graphs, use the
[equation native IR compiler](equation-native-ir.md). Its bound model contract
and recompilation check are separate from the compound primitive attributes
described here.

## Dense hardware words

`sc.dense_forward` lowers to `sc_dense_matrix_layer`, with one shared input
encoder per input and a separate row of weight encoders for each neuron.
Inputs are a vector of `ni` probabilities; weights are a row-major vector of
`nn * ni` probabilities. Each operand may use `rate` (signed Q8.8) or its own
signed `fixed<width,frac>` element type. A scalar is not a matrix and is not
broadcast. The Python `constant_f64_vec(values, ty)` accepts an element type
and infers the vector length, or accepts an explicit vector type whose length
must match.

Before the 16-bit encoder, each signed probability word is converted to
`floor(clamp(raw / 2**frac, 0, 1) * 65535)`. Intermediate products retain the
full declared width. The existing encoder compares its 16-bit LFSR state
strictly below that threshold; it is not a new random-number generator or a
claim of independent samples. Over a complete 65535-state LFSR period, the
realized fraction of ones is `max(T - 1, 0) / 65535`, where
`T = floor(clamp(raw / 2**frac, 0, 1) * 65535)`. Thus an input probability of
1 produces 65534 ones per period, not an all-ones stream; a threshold of 1
produces no ones. This retains the existing strict comparator. Finite windows
and the reset-prefix pipeline can differ from the full-period fraction.
Inputs and weights retain their separate binary points even when the neuron
word has another width.

State, leak, gain and current endpoints use signed words with eight fractional
bits. `dw` may be 10 through 64, so the fixed threshold 256 is representable.
Leak and gain must be matching `fixed<dw,8>` or raw `i<dw>` words (`rate` also
matches at `dw=16`). For a raw `i16` state coefficient, integer 256 means 1.0
in Q8, not a physical coefficient of 256; no implicit integer-to-Q8 scaling is
performed. Both current endpoints must fit the signed word; ascending
and descending ranges use signed division toward zero. Stream lengths must
fit a positive 32-bit count. Invalid shapes, formats, endpoints and absorbing
zero seeds are diagnostics from both graph verification and HDL emission.

Input seeds are `(input_seed + 7*i) mod 65536`; row-major weight seeds are
`(weight_seed + 13*(row*ni+i)) mod 65536`. Every generated seed must be nonzero.
The Python `seed_base` retains its existing mapping: input base is `seed_base`
and weight base is its 16-bit successor. Legacy direct RTL consumers retain
their seed defaults; generated IR selects a 16-bit probability bus independently
of the neuron word width.

The generated Dense circuit runs continuously. Encoders register their next
bits on each rising edge; the LIF neurons consume the previous registered bits
on that edge. `len` controls the existing running/done counter, not a pause or
reset of neuron state. An asynchronous reset restores all declared encoder
seeds and neuron state, including reset pulses between clock edges. This graph interface does not introduce an AXI input
commit handshake. The separate SHD board-validation interface is unchanged.

Python's default `y_max` is now 256, matching Rust's signed Q8 default and the
previous actual hardware mapping. The earlier Python default 65535 was stored
in text but ignored by the emitter. Explicit 65535 with `dw=16` now produces an
error; regenerate the graph with the intended signed current endpoint rather
than relying on truncation.

## Constant representation

Floating constants are scaled using their declared binary point and truncated
toward zero. Signed and unsigned integer constants already denote encoded raw
words. Scalar widths, vector element widths and vector lengths are preserved
in generated buses. Non-finite floats, overflow, malformed scalar/vector shapes
and invalid word geometry are rejected instead of being saturated or silently
narrowed to 16 bits. `rate` retains its Q8.8 storage; probability clipping occurs
at the encoder boundary.

When a raw integer constructor could otherwise be read as a float or another
integer variant, the printer uses `i64(N)` or `u64(N)`. In particular,
`sc.constant i64(1) : rate` means the raw Q8.8 word 1; legacy
`sc.constant 1 : rate` retains its historical real-valued meaning of 1.0,
encoded as 256. Both explicit tags preserve their constructor across text
round-trips and refuse malformed or overflowing literals.


## Clocked streams and packed words

`bitstream<N>` denotes one physical bit per clock with a positive observation
window of N clocks. Its length is temporal, not a 64-bit packed storage width.
`u<W>`, `i<W>` and `bool` denote packed words; vectors concatenate their element
buses. AND and XOR preserve the operand type and require identical shapes,
including identical temporal window lengths.

`sc.popcount %input : u64` counts the actual physical width of a packed word
combinationally. For a temporal stream, it accumulates samples present immediately
before rising clock edges over consecutive, non-overlapping windows. The completed
count becomes visible after the final edge and holds until the next completed
window. A vector of temporal streams counts all its lanes on each edge.
Asserting active-low reset clears phase, the partial sum and the held result
asynchronously, including between clock edges. The first window begins at the
first rising edge following reset release.

`sc.decode %input : rate`, exposed as Rust `ScGraphBuilder::decode` and Python
`ScGraphBuilder.decode`, performs the same observation and returns signed Q8.8
storage containing `floor(count * 256 / observations)`, from 0 through 256.
The denominator is the packed bus width, or the number of stream lanes times
its temporal window length. Counting and scaling preserve all intermediate bits;
the result is a normalised probability, not integer division of a count by its
window. A decoded Rate can feed another encoder without losing its binary point.
`div_const` retains unsigned integer division of a finite packed integer word,
returning the low 64 quotient bits after full-width division. It rejects a zero
divisor, invalid bus geometry and probability or temporal-stream operands.
Signed integer storage is interpreted as an unsigned bit pattern.
Count or decode a temporal stream before dividing its result. Integer division
does not perform fixed-point normalisation.

Encoders register their output bit on the rising edge. A downstream counter
on that same edge therefore consumes the preceding registered bit. The first
encoder-fed window includes the reset-defined zero prefix; subsequent windows
are contiguous. This pipeline latency also applies when a decoded result feeds
a new encoder. The encoder retains its existing deterministic 16-bit LFSR and
strict comparison; decoding does not establish independent random samples.

The Rust public builder, text parser/printer, Python bridge and direct
SystemVerilog emitter share these contracts. Primitive CIRCT lowering uses
HW/Comb/Seq operations, with signed probability clipping and asynchronous
register resets. Compound neuron lowering and complete compiler acceptance
remain separate from the primitive numerical checks.

Target resource reports remain compiler planning estimates. Temporal counting
includes the declared phase, accumulator and held-count storage before synthesis
optimization. Packed counting is combinational. LUT and critical-path heuristics
are not post-synthesis utilization or timing measurements; use synthesis and
post-route reports for platform capacity or clock-frequency decisions.


## Explicit scalar words

`ScGraphBuilder.word(WordOp)` in Rust and the Python `word_*` builder methods
construct typed scalar circuits. Both SystemVerilog and CIRCT emission implement
the same operations. A stored instruction retains its operands, result geometry,
rounding mode and saturation bounds:

```text
sc.graph @square_q8 {
  %0 = sc.input "x" : i16
  %1 = sc.word resize %0 : i32
  %2 = sc.word mul %1, %1 : i32
  %3 = sc.word round %2, 8, half_even : i32
  %4 = sc.word clamp %3, -32768, 32767 : i32
  %5 = sc.word resize %4 : i16
  %6 = sc.word bitcast %5 : fixed<16,8>
  sc.output "square" %6
}
```

Arithmetic requires identically typed raw integers (`iN` or `uN`). `add`, `sub`,
`mul` and `neg` retain the low N bits; they do not saturate. To preserve a full
product, resize both operands before multiplication. `div` truncates toward
zero for signed integers; `rem` has the numerator's sign. Both return zero for
a zero divisor. Signed minimum divided by minus one wraps to the same minimum.
Variable division uses an exact restoring circuit, including words wider than
64 bits; it does not approximate division using a rounded reciprocal.

`and`, `or`, `xor`, `not`, `shl`, `lshr` and signed-only `ashr` operate on bits.
Shift counts are unsigned physical words, even when the input type is signed.
A count at least N yields zero for logical shifts or sign fill for `ashr`.
`cmp_eq`, `cmp_ne`, `cmp_lt`, `cmp_le`, `cmp_gt`, `cmp_ge` produce `bool`; ordering
uses operand signedness. Python spells these `word_compare("lt", lhs, rhs)`.

| Operation | Python method | Meaning |
|---|---|---|
| `resize` | `word_resize(input, ty)` | Sign/zero extend according to the input type, or discard high bits |
| `bitcast` | `word_bitcast(input, ty)` | Preserve physical width and bits, changing scalar interpretation |
| `select` | `word_select(condition, true, false, ty)` | Boolean condition, identical scalar branch types |
| `round` | `word_round(input, shift, mode, ty)` | Divide by `2**shift` with explicit rounding and no intermediate overflow |
| `clamp` | `word_clamp(input, min, max, ty)` | Inclusive saturation at ordered, representable signed-64-bit literal bounds |

A bitcast does not rescale fixed-point data. Arithmetic on `fixed<width,frac>`
or `rate` therefore requires a bitcast to an integer first, followed by explicit
rescaling at the source-defined boundary. Temporal bitstreams and vectors are
not scalar words. Boolean conditions must be `bool`, not a numeric 0/1 word.

Rounding names are `floor`, `toward_zero`, `half_up` (ties toward positive
infinity), `half_away` (ties away from zero), and `half_even`. A zero shift is
identity; otherwise the shift is less than N. `clamp` is separate from arithmetic
so that the equation frontend can preserve the original commit boundaries.
These word operations provide arithmetic circuits; they do not by themselves
establish arbitrary equation, integration-method or state-machine lowering.
Resource reports remain planning heuristics; use synthesis and post-route
reports for actual FPGA utilization and timing.

Module and port names must be ASCII HDL identifiers and cannot be language
keywords. Ports are unique across inputs and outputs. `clk`, `rst_n`, `sc_*`,
`encode_*`, `decode_*`, `dense_*`, `u_*`, and `v`/`c` followed by a digit are reserved for
compiler-generated signals. Verification and both emitters refuse collisions
with a diagnostic; public names are never silently changed.

Clamp bounds use the signed-64-bit literal interface even for a wider data
word. Bounds above `2**63 - 1` or below `-2**63` cannot be supplied; the Python
native boundary raises `OverflowError` before appending an operation. A wide
word can still be clamped to a representable interval. Wider bound literals
require a future width-tagged constant interface.

All declared input, constant and word-result types are checked even when the
value is unused. Widths and vector counts must be positive, fixed-point
fractions must be smaller than their width, and the total physical width must
fit positive Verilog indexing. Vector nesting is limited to 64 wrappers and
checked before either text constructor recurses. Exhausted parsed value IDs
produce a parse diagnostic instead of wrapping the parser allocator.

## Native identifier allocation

Rust `ScGraph::try_fresh_id()` returns `Result<ValueId, ValueIdExhausted>`.
It reserves an identifier without appending an operation. Identifiers run from
zero through `u32::MAX - 1`; the counter value `u32::MAX` represents exhaustion
and is never allocated. A sparse graph loaded from text can reach this boundary
without containing billions of operations. Refusal leaves the counter and every
operation unchanged, so the graph remains printable, verifiable and emit-ready.

The existing `ScGraph::fresh_id()` and builder method signatures are retained.
`fresh_id()` panics with `SC IR value ID space exhausted` before mutation if
no identifier remains, consistently in debug and release builds. Callers that
extend externally supplied graphs should use `try_fresh_id()` and handle its
error. A panic-aborting application cannot recover from the legacy call.
The Python graph API does not expose either allocator or a builder resumed from
a parsed graph; this addition is a native Rust API, not a new Python method.

Division by the constant one preserves the unsigned physical input bits, even
for a signed constant. Both backends lower this identity to explicit zero
extension or low-bit extraction for the destination width. This also applies
to the scaling division in a one-tick decode; it does not change reset or
window-completion timing.

## Scalar register transactions

Rust `ScGraphBuilder::state(initial, ty)` and Python `state(initial, ty)`
declare current register storage. The initial value must be an earlier scalar
constant of exactly the same type. Supported storage types are `bool`, `rate`,
`fixed<width,frac>`, `iN` and `uN`, with widths from 1 to 64 bits. Floating reset
constants use the existing quantisation toward zero; integer constants are
already raw words. Wider arithmetic intermediates remain available through
explicit word operations and must be converted to the state type before storage.

Each register requires exactly one `store(state, next, enable)` binding.
Python defaults enable to `None`; Rust uses `Option<ValueId>`. A present control
must be `bool`: false holds the current value and true permits the update.
The next value must match the complete state type, including binary point and
signedness. A store returns a unique terminator identifier, which cannot feed
an output, arithmetic, another store or a state declaration. Outputs also remain
non-value terminators. Missing or duplicate drivers, non-State targets and
malformed types are errors from verification and both emitters.

All registers capture their next values simultaneously on the rising edge of
`clk`. Each expression observes the preceding committed state until that edge;
a two-state swap therefore exchanges both values without sequential overwrite.
Active-low `rst_n` restores every initial word asynchronously, including pulses
between edges, and takes priority over enable. Direct SystemVerilog uses
nonblocking assignments; CIRCT uses `seq.firreg` and an explicit hold mux when
enabled. Register feedback does not introduce combinational SSA cycles.

```python
from sc_neurocore_engine.ir import ScGraphBuilder

b = ScGraphBuilder("accumulator")
enabled = b.input("enabled", "bool")
zero = b.constant_i64(0, "i16")
one = b.constant_i64(1, "i16")
current = b.state(zero, "i16")
next_value = b.word_binary("add", current, one, "i16")
b.store(current, next_value, enabled)
b.output("count", current)
graph = b.build()
assert graph.verify() is None
```

The lossless text records every binding and the absence of enable explicitly:

```text
%0 = sc.constant -320 : fixed<16,8>
%1 = sc.state %0 : fixed<16,8>
%2 = sc.store %1, %1, enable=none
```

Resource planning adds each declared register's width to the flip-flop estimate
and one width of mux logic for an enabled store. These are static estimates;
board resource usage, clock frequency and routed timing require actual synthesis
and implementation reports. Register transactions provide an IR building block;
they do not establish general equation, integrator, event or RNG lowering.
