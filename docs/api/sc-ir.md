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
