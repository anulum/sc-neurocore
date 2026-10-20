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
