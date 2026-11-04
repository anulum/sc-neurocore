<!-- SPDX-License-Identifier: AGPL-3.0-or-later -->
<!-- Commercial license available -->
<!-- © Concepts 1996–2026 Miroslav Šotek. All rights reserved. -->
<!-- © Code 2020–2026 Miroslav Šotek. All rights reserved. -->
<!-- ORCID: 0009-0009-3560-0851 -->
<!-- Contact: www.anulum.li | protoscience@anulum.li -->
<!-- SC-NeuroCore — Per-function hardware measurements -->

# Per-function hardware measurements

The standalone function emitter isolates the equation compiler's existing
fixed-point arithmetic in a combinational module. The source-checkout synthesis
command measures that module with Yosys and proves its mapped implementation
against the emitted reference. Each measurement belongs to one function,
lowering and signed fixed-point format.

## Emit a function

```python
from sc_neurocore.compiler import emit_function_unit_verilog, function_unit_functions

functions = function_unit_functions("v2")
rtl = emit_function_unit_verilog(
    "sin", data_width=16, fraction=8, function_lowering="v2"
)
```

The module is named `function_unit`. Its signed ports are `argument0`,
`argument1` for binary functions, and `result`. Both binary operands remain
independent inputs. There is no clock, reset or neuron-update state. Zero-cycle
latency describes the combinational interface; it does not establish a clock
frequency or propagation delay.

Select `v2` or `lut-v1` explicitly when comparing their implementations.
`function_unit_functions(lowering)` returns that lowering's canonical supported
names. `expit` is an alias for `sigmoid`; aliases do not add report rows.

## Measure one implementation

Run from the source checkout with the project dependencies, Yosys and its
associated ABC executable available:

```bash
python -m tools.function_synthesis \
  --cache .cache/function-synthesis \
  --output .cache/function-synthesis/sin-diagnostic.json \
  --synthesis-timeout 240 --proof-timeout 90 \
  measure sin --data-width 16 --fraction 8 --lowering v2
```

`--synthesis-timeout` sets the mapping command's elapsed-time budget, which
defaults to 240 seconds. Large mappings can require a larger budget; this changes
neither the emitted arithmetic nor the mapping target. `--proof-timeout` sets
each native proof command's upper bound, capped at 90 seconds. A complete proof
can require several commands. Mapping equivalence has one overall 3,300-second
budget across composition, complete-domain fallback and final validation.
The Python `check_mapped_equivalence(..., cohort_seconds=...)` entry point
accepts a smaller overall budget; switching proof methods never resets it.

The target is Xilinx 7-series (`xc7`), with I/O pads and clock buffers excluded
from function area. Hardware values include the actual mapped cell inventory,
DSP blocks and separately counted source table bits. When native statistics
separate primitive module instances from ordinary cells, the total includes
both and must agree with every reported cell type and the physical netlist.
Table bits describe the
emitted tables; they do not imply that the target inferred that capacity as
block RAM. A single measurement is a diagnostic and cannot complete the matrix.

The acquisition completed on 4 November 2026 covered all 270 supported cases:
228 produced admitted measurements, 19 had rejected or incomplete proof
evidence, and 23 reached a native deadline. Another 102 combinations are
unsupported by their selected lowering, giving 372 dispositions across both
lowerings and all six formats. This is the current acquisition status, with
complete matrix acceptance still open.

The individual diagnostic reports retain those failure dispositions. A
cache-only matrix report reads successful entries and therefore reports the
42 unsuccessful cases as missing; it does not replay failed attempts or
convert them into measurements. These mapped resource counts are synthesis
results, rather than measured FPGA latency or energy.

Tool inspection runs Yosys and identifies its native process executable through
Linux procfs, including when the requested command is a launcher script. All
subsequent commands and SHA-256 bindings use that actual binary and its sibling
ABC executable. The launcher cannot hide a later change to the native binary.
Native commands receive a minimal environment and refuse `LD_*` loader overrides.
Interrupted callers terminate and reap their owned native process group; an
outer deadline also supervises GNU timeout itself. Executable and library hashes
are checked before and after each native command. These checks assume a trusted
operator and installed SDK: a privileged actor coherently replacing and restoring
the installation during execution is outside this custody model. Hashes and
retained logs provide reproducibility and corruption detection, rather than
cryptographic attestation of execution.

The cache retains the emitted RTL, mapping commands, mapped netlist, native
logs and complete equivalence evidence. Reuse requires matching compiler and
proof sources, executables, technology-library contents and retained artifacts.
Readers preserve the original measurement time. Failed or inconclusive native
proofs do not become successful measurements.

Successful acquisitions retain every original artifact in a single lossless
compressed representation, with the original byte counts, SHA-256 identities
and file modes. Cache admission restores a bounded temporary view and runs the
same complete proof readers against the original bytes. Original acquisition
times remain unchanged. For direct inspection of a cached result, use
`with result.open_artifacts() as directory:`; the original filenames are
available inside that context, and the temporary view is removed on exit.
Interrupted or corrupt retention never publishes a successful cache entry.
For a constrained storage allocation, `retain_artifacts(directory, byte_budget=...)`
can lower the admitted total original artifact bytes. Its default and maximum
remain 1 GiB; a larger requested limit is refused. The complete native evidence
and lossless format are unchanged.
Restored views are created beside their cache entry so that they remain on the
same storage volume and inside its disk accounting. Legacy uncompressed cache
entries without a retention manifest remain readable: they require regular,
unlinked files and the same source and complete native proof validation. Only
the compressed format enforces an exact whole-directory inventory; extra
unreferenced regular files in a legacy entry confer no proof credit.

The proof method records the actual successful path: `complete-aiger` for
a complete native comparison of the reference and mapped exports,
`compositional` for DSP component proofs, `complete-bdd` for a direct
whole-original-function native BDD proof, or `complete-bdd-shannon` for a
complete Shannon-partitioned BDD proof.
The compositional method binds the original RTL to its synthesis checkpoints,
checks exact word arithmetic and each configured DSP cell's complete contract,
then proves the complete residual graph with native equivalence. That residual
compares every connected DSP input and every original result bit, with all
original arguments retained and the independent primitive-cut values universally
quantified. Every component must pass before the complete proof record can be
created.

When a DSP compositional attempt cannot establish complete equality, the
`complete-bdd` method independently compares the whole original function with
the whole mapped implementation. Every original argument bit remains independent
and every original result bit is compared. This method has no auxiliary
primitive-cut inputs.

Its complete native exports use lossless compressed RTLIL checkpoints so that
each export command retains its ninety-second ceiling. Preparation, generic
technology mapping and AIG serialization run as separate commands for each
of the reference and physical graphs. Both the prepared hierarchy and the
mapped checkpoint are retained and checked for complete gzip integrity;
the checkpoint record uses schema version 2. The six commands share the same
whole-export allowance. Only private internal
identifiers are renamed for serialization; the original RTL, physical mapping,
cell functions, parameters, connections and public ports remain unchanged.
Native normalization is checked before forming the OR of every output
difference. ABC constructs global BDDs with fixed variable ordering and a
bounded live-node budget. A measurement requires the actual complete difference
to reduce to constant zero. A nonzero result or incomplete native protocol
receives no proof credit. Exhausting a global node or time limit can start the
complete input partition described below; that limit alone never proves equality. Original failed component
artifacts remain retained alongside the successful direct proof.

Readback revalidates the original source, physical clone, export scripts,
compressed-stream integrity, native commands, complete interfaces and final
native constant. It preserves the original acquisition time.
Each native normalization round trip explicitly compares inputs and outputs
by position with ABC's `cec -n`. The original reference and mapped exports
first require identical named bit order, so the positional all-output miter
retains that original function domain.

Each DSP arithmetic contract compares all 192 operand bits and 26 independently
variable auxiliary controls against all 144 arithmetic and cascade output bits.
The admitted combinational profiles require zero registers and known zero
auxiliary controls in the actual mapped graph. The independent contract proves
that those controls cannot change the compared arithmetic in these profiles.
A graph with auxiliary pins must supply all 26 auxiliary inputs. Unconnected
status outputs may be absent; every connected port still requires its exact
width and direction. The word checker admits five configurations: direct A/B
for modes `0000101`, `0010101` and `0110101`, and exactly one cascaded operand
for mode `1010101`. Other configurations remain rejected even when their
standalone native arithmetic contract can be proved.

When native mapping adds explicit controls and unused cascade or status pins, a separate
source clone receives the same qualified zero controls and fresh independent
output wires. Every added output must be unobserved by original
ports or cell inputs and have exactly one driver. The residual proof includes
all connected primitive inputs and all original results; its independent cut
values include the added outputs. The retained derivation binds the complete
original source, physical graph and derived clone. Cache readback recomputes it
from the original graphs before admitting the proof.

Source normalization first compares exact expressions for every original result
bit. A multiplexer bit depends on its selector and its two connected data bits;
unused bus lanes and the bit's position do not change that scalar function.
Port geometry, the declared bus width and supported parameters are validated
before constructing these identities. This admits native width reduction while
preserving changes to either data input or the selector. Different expression
identities still require the complete native source proof.

The two independent source exports run concurrently. A measurement can therefore
use two native processes; each retains its own ninety-second command ceiling.
Both exports must complete before their equivalence proof starts.

Native processes use the C locale and a fixed wide readline display. This keeps
ABC's complete command echo available to Yosys's completion protocol even when
private artifact paths are long. It changes neither the mapping commands nor
their deadlines.

Complete AIG comparisons preserve every independent named input bit. The
physical export may reorder those bits only when the native checker confirms
that it matches by name. The original output-bit order stays exact. Missing,
repeated or changed bit names are refused.

These methods establish equality with the emitted fixed-point implementation.
They do not establish equality with a real-valued mathematical function; the
separate numerical assessment describes that approximation.
The standalone `check_mapped_equivalence` entry point also accepts legacy bare
AIG pairs; in that form it establishes equality of those complete named export
functions. A hardware measurement additionally requires the original mapped
netlist, native statistics and source-bound export protocol validated by
`read_function_synthesis_measurement`. A bare-pair verdict alone supplies no
physical resource evidence. No method
admits sampled inputs, partial outputs or an inconclusive solver result as a
complete mapping proof.

Native JSON exports retain the complete function module and its directly
referenced cell models. Unrelated technology-library modules are excluded from
each export; the inspected toolchain separately binds every library file.
The native selection explicitly retains directly referenced blackbox and
whitebox definitions, including their port directions. Unrelated library
processes remain outside the export. Exports copy the live native design and
then restore it in full. They preserve
the physical mapping commands, cell inventory, ports, nets and parameters.

If one global BDD reaches its existing node or time limit, the compiler can
complete the proof by partitioning the original input domain. The public proof
method is then `complete-bdd-shannon` (complete BDD, Shannon-partitioned).
Each terminal cofactor compares the OR of every original output difference,
with all remaining inputs independent. A separate verifier checks the actual
cofactor assignments against their declared prefixes and verifies that their
intervals cover the entire original domain exactly once. Missing, duplicate,
overlapping or swapped branches invalidate the proof.

The retained record states the deterministic split-variable order and its
rationale, every native command and original acquisition time, source and
toolchain identities, and the complete artifact hashes. Every BDD keeps the
same one-million-live-node and ninety-second limits; the whole partition has
at most 3,300 seconds and 256 nodes. The whole-function API subtracts the time
already spent exporting, normalizing and attempting the global proof before
starting the partition. Both proof APIs accept a smaller `cohort_seconds`
allowance; every native command must fit its remaining
allowance. Successful records are written only after reconstructing the proof
and checking the elapsed allowance again; the partition's elapsed time includes
that reconstruction. Exhaustion leaves the proof
incomplete and contributes no measured case. Input partitioning does not
change the mapped hardware, numerical error measurement or unsupported-case
inventory.

## Read or acquire the comparison matrix

Reading a report uses retained successes and leaves missing measurements explicit:

```bash
python -m tools.function_synthesis \
  --cache .cache/function-synthesis \
  --output .cache/function-synthesis/comparison.json \
  report
```

Use `report --measure-missing` to request the missing mappings and proofs.
Choose a fresh output path for each report; the command refuses to overwrite
an existing file.

The matrix covers both lowerings at `(data_width, fraction)` values `(16, 8)`,
`(16, 12)`, `(32, 16)`, `(18, 9)`, `(32, 24)` and `(16, 15)`. Every canonical
function has a disposition for each lowering and format. Unsupported legacy
functions carry unavailable hardware values. Missing, timed-out, failed or
rejected evidence keeps the report incomplete and produces a nonzero exit.
Exit status 1 accompanies an incomplete report or unsupported diagnostic;
status 2 identifies a setup or output error, and status 3 identifies an
unexpected internal failure. Internal failures do not publish a report.

Each measured row includes a separate numerical domain and error assessment
for the selected lowering. Shared names and formats do not establish equal
accuracy. For `lut-v1` rows of `floor`, `ceil`, `trunc` and `sign`, the emitted
arithmetic is shared with `v2`. Their numerical assessment explicitly records
`assessment_basis="shared_exact_arithmetic"` and its actual `v2` assessor lowering;
it is not a lookup-table accuracy claim. Compare area together with these assessments. Cell counts establish
neither device timing, power, throughput nor a numerical speedup. See
[Function unit numerical error](../validation/function_units_v2_accuracy.md)
for the numerical assessment contract.
