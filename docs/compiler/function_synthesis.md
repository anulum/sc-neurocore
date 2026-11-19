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

The `v2` implementation of `atan2` forms the ratio of the smaller magnitude to
the larger. Restoring division computes only the 30 fractional bits needed
by the interpolation table. The operand order and signs determine the final
angle, including the exact axis cases.

Unary arctangent shifts its fixed-point magnitude directly on [0, 1]. For
larger magnitudes, bounded restoring division computes its reciprocal. Both
paths use one table interpolation and preserve the reference truncation.

The inverse sine and cosine use the same fractional division after ordering
the magnitude and its complementary square root. The integer square root
consumes the radicand two bits at a time and bounds each intermediate
remainder. The operations preserve the fixed-point reference's truncation,
saturation and exact endpoints.

Cube root uses shifts and sums for the original Q30 coefficient and bit masks
for the fixed-point remainders. Where the input format and logarithm table
guarantee a negative scale, it uses a rounded right shift. The other formats
use the general scaling circuit.

The relative exponential uses a polynomial series near zero. When the input
bounds and output scale permit a restoring divider limited to the result width,
the circuit emits the saturated positive tail directly. The general path uses
one shared signed divider. Both paths preserve truncation and final rounding.

Tangent rounds the sine-to-cosine ratio in magnitude before restoring its
sign. It caps the dividend at the signed output rail times the divisor,
so the restoring divider only computes the quotient bits the output can use.
Inputs whose computed cosine is zero retain the original pole result.

Reciprocal square root uses a bounded quotient followed by an integer square
root. It skips quotient digits above the output range and checks saturation
from the original input. The divider and square-root remainders locate the
exact rounding midpoint. For wide fractions or long skipped prefixes, both
stages reuse each trial subtraction's borrow bit to decide whether the trial
fits.

The decimal logarithm computes the leading exponent separately from the
bounded base-two mantissa. Their contributions use the same fixed-point
coefficient and preserve the original rounding and saturation.

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

The comparison at commit `f50004f5dd1c80008d8adca3944114a23b7a09b9`
contains 372 cases across both lowerings and all six formats. Using Yosys 0.70+post and its associated ABC, all 270 supported cases
produced source-bound Xilinx 7-series measurements and passed equivalence for the
entire mapped interface. The other 102 cases are explicitly unsupported by their
selected lowering. The cache-only report revalidates the successful measurements
and preserves their acquisition times. These figures describe mapped resources.
The routed timing experiment below tests a separate constraint on one device.

Tool inspection runs Yosys and identifies its native process executable through
Linux procfs, including when the requested command is a launcher script. All
subsequent commands and SHA-256 bindings use that actual binary and its sibling
ABC executable. The launcher cannot hide a later change to the native binary.
Native commands receive a minimal environment and refuse loader injection
settings such as `LD_PRELOAD` and `LD_AUDIT`. The caller may have
`LD_LIBRARY_PATH` set for its Python installation; that setting is never passed
to a native child.
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
Restored cache views are created in the cache root, outside the published
success directory. They remain on the same storage volume and inside its disk
accounting, and a forcibly interrupted reader leaves the published directory
untouched. Legacy uncompressed cache entries without a retention manifest remain
readable: they require regular, unlinked files and the same source and complete
native proof validation. Only the compressed format enforces an exact
whole-directory inventory. Extra unreferenced regular files in a legacy entry
confer no proof credit.

The proof method records the actual successful path: `complete-aiger` for
a complete native comparison of the reference and mapped exports,
`complete-aiger-gia` when the direct checker reaches its watchdog and a
separate GIA proof establishes equality of those same original exports,
`compositional` for DSP component proofs, `complete-bdd` for a direct
whole-original-function native BDD proof, or `complete-bdd-shannon` for a
complete Shannon-partitioned BDD proof.
The compositional method binds the original RTL to its synthesis checkpoints,
checks exact word arithmetic and each configured DSP cell's complete contract,
then proves two adjacent complete residuals with native equivalence: the
post-DSP graph against a retained pre-LUT checkpoint, and that same checkpoint
against the unchanged physical mapping. Only CARRY4 primitives are expanded in
the intermediate checkpoint; DSP instances and the physical mapping flow remain
unchanged. Each residual compares every connected DSP input and every original
result bit, with all original arguments retained and its independent
primitive-cut values universally quantified. The shared intermediate module,
rather than matching names of cut values, connects the two function proofs.
Both components run in parallel under one shared deadline. Each must pass
before the complete proof record can be created.

The intermediate export enumerates generated generic wire and cell names while
preserving primary port names and DSP instance identities. It retains all cell
attributes, parameters, connections and signed annotations. Oversized native JSON
is first compacted by removing whitespace outside strings. If it still exceeds
the 64 MiB parser limit, a bounded format stores each repeated native source
location once and restores every original value before building the proof
graph. Both formats retain the byte-for-byte native original and its SHA-256.
The reader reproduces the location table from the original. Oversized derived
cuts are checked against their exact reconstructed native JSON bytes without
decoding them again.

The read-only consumer reconstructs both exact cut graphs, including the
primitive-output alias partitions, controls, original output edges and export
commands, from the retained original checkpoints. It then validates both native
proofs without rerunning tools or changing acquisition timestamps. The native
pre-LUT mapping keeps a compressed RTLIL checkpoint before expanding carry
cells. With a variable divisor or at least 500 coarse cells,
it retains checkpoints at the XC7 coarse, fine and cell-mapping boundaries
so each phase has its own
bounded native command. Each residual cut also uses lossless RTLIL handoffs. The independent
word and physical exports run in parallel. Word graphs with
division between two variable operands replay the corresponding XC7 mapping
phases before gate export. A constant dividend can still require this route
when its divisor varies. The staged export gives the checker more similar divider
structures while preserving the complete cut interface. Physical lowering uses
a separate command to expand library primitives before generic optimization.
When the mapped LUTs can expand to at least 65,536 decision nodes, the physical
proof graph folds constants and removes unused logic. At 524,288 estimated
decision nodes, constant folding and width reduction run as separate native
commands. Shortening private
names and omitting cell and wire source locations starts before folding.
Later derived RTLIL handoffs retain the functional top. Native structural
checking and BLIF export use the same checkpoint
in separate bounded commands. The BLIF export preserves all named proof ports
and omits aliases without fanout. Native validation checks that every original
output is driven by its declared inputs and defined logic. The inspected ABC
binary hashes the BLIF structure into a named AIG. The physical synthesis
and resource accounting stay unchanged. Every command retains its native
deadline and shares the original proof cohort budget. The reader reconstructs
the scripts, checks gzip integrity and binds the cut, commands, logs, BLIF
when present, and final AIG to the proof. Altered cuts cannot reuse a previously
successful native verdict.

DSP mapping can retain a multiplication as ordinary logic. The word proof
checks the retained cell's parameters, directions and every operand and result
bit against the original before proving the complete product.

When DSP mapping absorbs adjacent additions or subtractions into a product,
the word checker proves the connected arithmetic component jointly. Every
externally consumed root keeps its complete declared word, including high and
sign bits, and is compared in `Z/(2^w Z)` after interpreting each intermediate
width, signed extension, truncation and slice. Frontier values require inductive
driver correspondence from original inputs and previously proved functions;
shared signal names alone establish no equality. The checker rejects cycles,
state and unaccounted logic reaching any original output. Globally unobserved
arithmetic receives an explicit disposition and remains in the actual resource
counts.

If a complete root has a nonzero normalized difference, acquisition retains
`group-refusal.json` with the full residue and a separating Boolean assignment
when the expression contains only Boolean atoms. That assignment concerns the
independent local frontier; it does not claim reachability from primary inputs.
A remaining floor expression can prevent this algebra from establishing equality,
so a nonzero normal form alone is recorded as an unproved identity.

Joint identities use `sc-neurocore.grouped-word-equivalence.v1` inside a version 2
word record. The complete two-stage mapping chain uses
`sc-neurocore.function-composition.v3`. `products_proven` counts individual product proofs only;
`products_covered_jointly` records products whose component was proved as a whole.
When a grouped proof is needed, every disappeared multiplication receives a
group certificate, including disconnected products which could otherwise use
the individual path. This accounting limitation preserves the proof strength;
it does not create additional individual-product certificates.
Components containing an absorbed addition or subtraction retain their connected
arithmetic. Other products start as singleton groups, keeping unchanged outside
functions in the structural induction. Groups still merge when their actual
mapped cones overlap.
Every surviving function retains its checked pin correspondence. The complete
source dependency order and each group's explicit outside frontier establish
the structural induction. Cell accounting uses the disjoint mapped cones
recorded before polynomial evaluation, then independently checks every actual
output cone. Public output ports are compared with the proved anchors; a port
can supply an anchor only when no shared name or surviving pin already supplies
one. Named ports must still match their declared connections.
Readers independently reconstruct the partition, complete roots, frontier,
arithmetic parameters and global cell accounting. Semantic certificates remain
deterministic under JSON mapping reordering; the enclosing acquisition retains
the separate exact raw-file identities.

Producer and reader work share the same enclosing 3,300-second wall deadline.
Individual-product proofs also inherit that deadline and check it during
materialised algebra work and before publication. Both word-proof paths require
every additional mapped function to have an arithmetic disposition.
The unchanged 100,000-term guard applies to every materialised expression on
both paths. Grouped proofs retain a separate ledger recording operation counts,
term occurrences, peak expression size and actual elapsed time. Cumulative term
occurrences are measurement evidence, rather than a new complexity limit.
Exhausted budgets cannot become equivalent verdicts or be relabelled as
certificate mismatches.

The residual checker first constructs the OR of every original output
difference. If its GIA solver explicitly times out with an undecided verdict,
a second CEC command compares the two complete normalized output vectors.
An undecided result with a recorded empty reduction and zero remaining nodes
also selects that complete comparison, even when the native log omits the
timeout line. The paired comparison can prove equality with its explicit
equivalent verdict.
Both models must retain identical named input and output order, and each
normalization must independently prove its round trip. The second strategy
requires an explicit equivalent verdict and records
`complete-paired-output-cec-after-undecided-gia` as the actual proof engine.
The original undecided transcript and both complete commands remain retained.
If paired GIA also reports a recognized bounded undecided result, ABC's
Ivy solver checks the existing scalar difference miter. This last check
requires an explicit UNSAT verdict within sixty seconds and retains the
command and transcript in a version 4 record. Negative, erroneous or
malformed verdicts stop the proof. All checks share the original
3,300-second budget, and the paired command keeps its ninety-second ceiling.
When the operating-system deadline ends a native comparison before it emits
a verdict, its original command and timeout transcript are retained. The next
complete comparison receives only the remaining shared budget.
An elapsed Ivy comparison can select bounded comparisons of original output
groups. The reader checks the unchanged binary graphs, matching structural
identities and phases, and coverage of every original output. Identical
ordered pairs require one comparison. Each group retains every original input
and all gates reachable from its selected outputs, with the original literal
phases. The reader reconstructs the same binary graph from the retained
complete models. A group that exhausts its native deadline is split into
disjoint output halves. A two-output subgroup is compared one output at a
time. A single undecided output
can be compared separately for every original argument sign combination.
For a single argument, a sign region that exhausts its deadline can be split
at the highest original byte boundary below the sign and then at the bit
immediately below the sign, if distinct. Each split requires affirmative
native equivalence for both bit values, with every unassigned input
independent. The retained record binds every group, cofactor command and
native verdict to the complete models under the same deadlines.

The Yosys 0.33 SDK uses ABC's two-circuit `&cec` mode for the paired
comparison. Its older `cec` implementation aborts on a retained complete
Hypot model pair; `&cec` proves the same ordered models without that abort.
The original singleton result, complete vectors, explicit final verdict and
native deadlines remain mandatory. Newer SDKs retain the `cec` paired command.

The legacy ABC executable with SHA-256
`001a99fb8648794076dd238e42f3bd8c5112a4b8bd89660883325a7e0d006043`
can report false equivalence for phase-complemented outputs with `&cec -y`.
That command on this build is not an admissible proof engine.

Every use of `&cec -y`, on any build, requires three phase-flip negative
controls with the same executable and solver configuration: complement the
first, middle and last compared output bits in separate candidate copies,
preserving every original input, output and their order. All three controls
must produce explicit non-equivalence before a positive result receives proof
credit. A timeout, an undecided result or a zero exit status alone cannot pass
this requirement. Retain the commands, binary hash, complete original and
mutated graphs, and native logs. These controls do not qualify the defective
legacy build.

The Python `prove_gia_equivalence(..., singleton_seconds=...)` entry point can
request a smaller internal singleton allowance. Its default is unchanged;
an explicit value must be an integer from one to the native ceiling. The
normalization and paired commands keep their original budgets. A nondefault
choice is bound to the singleton command and retained in the record.
The compositional residual producer uses a one-second singleton probe. An
undecided probe still requires the complete paired comparison, with its original
budget and every original input and output retained.
Primitive residual records report the engine that actually proved equality.

When a DSP compositional attempt cannot establish complete equality, the
`complete-bdd` method independently compares the whole original function with
the whole mapped implementation. Every original argument bit remains independent
and every original result bit is compared. This method has no auxiliary
primitive-cut inputs.

Its complete native exports keep the ninety-second ceiling per Yosys command.
The standard route separates preparation, generic technology mapping and AIG
serialization for the reference and physical graphs. Both RTLIL checkpoints
in each graph retain their gzip integrity checks and use checkpoint schema
version 2. When the mapped LUTs can expand to at least 524,288 decision nodes,
the export retains the complete top module at each native RTLIL handoff.
A verified timeout in a direct export can also start the staged export with
the remaining time from the same proof allowance. Readback verifies the
original timeout, commands and artifacts. The physical route expands the
inspected primitive library and
simplifies word multiplexers before gate expansion, preserving native
don't-care behavior. It writes BLIF with native aliases. Both graphs must
pass an independent Yosys read and structural check before ABC writes their
named AIGs. The export and its reader trace every original output back to its
drivers. Outputs that depend on an undriven or undefined name are refused
before an implicit constant can change the comparison. Checkpoint schema
version 4 binds the RTLIL snapshots, BLIF files, native commands and both
complete AIGs. The large RTLIL snapshots are compressed after the native
tools finish using the original files. Readback validates both their decoded
bytes and the stored representation. Reference and physical exports run
independently under one whole-export allowance. The original RTL, mapped
physical artifact and public ports remain unchanged.
Native normalization is checked before forming the OR of every output
difference. ABC constructs global BDDs with fixed variable ordering and a
bounded live-node budget. A measurement requires the actual complete difference
to reduce to constant zero. A nonzero result or incomplete native protocol
receives no proof credit. ABC can report a bounded construction abort without
a node count if an individual BDD operation hits its limit. A verified abort or
deadline can start the complete input partition described below. A complete BDD proof binds the
failed component artifacts alongside the direct proof. Large RTLIL, BLIF and
JSON files from the earlier attempt are retained losslessly. Readback checks
their decoded content without restoring those files to disk. The 1 GiB
aggregate cache restoration limit still applies to the complete cache payload.

Readback revalidates the original source, physical clone, export scripts,
checkpoint integrity, native commands, complete interfaces and final native
constant. It preserves the original acquisition time.
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

## Timing after placement and routing

Vivado 2025.2 (build 6299465) implemented the Q8.8 sine and square-root
units for the Zynq-7000 `xc7z020clg484-1` using the compiler at commit
`f50004f5dd1c80008d8adca3944114a23b7a09b9`. The out-of-context harness adds a
launch register and a capture register around each combinational unit and
constrains their shared clock to 10 ns (100 MHz). The timing scope is the
internal register-to-register paths. Setup and hold each meet the target
when their worst slack is non-negative.

| Function | Lowering | Slice LUTs | Slice registers | DSPs | Block RAM tiles | Setup slack (ns) | Hold slack (ns) | Timing at 100 MHz |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Sine | `lut-v1` | 9 | 16 | 0 | 0.5 | 3.285 | 0.316 | Met |
| Sine | `v2` | 656 | 32 | 8 | 0 | -27.939 | 1.799 | Missed |
| Square root | `lut-v1` | 16 | 26 | 0 | 0 | 6.511 | 0.228 | Met |
| Square root | `v2` | 1693 | 32 | 0 | 0 | -100.332 | 2.199 | Missed |

The routed functional netlist can be simulated against the corresponding
original RTL across all 65,536 input codes. Compare the result after it has
passed through both register stages. Reported utilisation counts include the
wrapper.
The I/O delay constraints below are example budgets for this standalone harness.
Timing and energy on a particular board require a board-level design and
instrumented measurements.

To reproduce a row, use an empty directory and run the following Python with
SC-NeuroCore installed. Change the function and lowering to select another row.

```python
from pathlib import Path

from sc_neurocore.compiler import emit_function_unit_verilog

rtl = emit_function_unit_verilog(
    "sqrt", data_width=16, fraction=8, function_lowering="v2"
)
Path("function.v").write_bytes(rtl.encode("utf-8"))
```

Save the register wrapper as `boundary.v`.

```verilog
module function_timing (
    input wire clk,
    input wire signed [15:0] argument,
    output reg signed [15:0] result
);
    (* DONT_TOUCH = "TRUE" *) reg signed [15:0] input_word;
    wire signed [15:0] output_word;

    function_unit dut (.argument0(input_word), .result(output_word));

    always @(posedge clk) begin
        input_word <= argument;
        result <= output_word;
    end
endmodule
```

Save the clock and I/O constraints as `timing.xdc`. The clock source is a
global clock buffer site on this part.

```tcl
create_clock -name function_clock -period 10 [get_ports clk]
set_property HD.CLK_SRC BUFGCTRL_X0Y16 [get_ports clk]
set_input_delay -clock function_clock -max 2 [get_ports {argument[*]}]
set_input_delay -clock function_clock -min 0 [get_ports {argument[*]}]
set_output_delay -clock function_clock -max 2 [get_ports {result[*]}]
set_output_delay -clock function_clock -min 0 [get_ports {result[*]}]
```

Save these Vivado batch commands as `route.tcl`.

```tcl
if {[llength $argv] != 3} { error "usage: part, period in ns, and thread count" }
lassign $argv part period threads
set_param general.maxThreads $threads
if {[llength [get_parts -quiet $part]] != 1} { error "part is not installed: $part" }
read_verilog function.v
read_verilog boundary.v
read_xdc timing.xdc
synth_design -top function_timing -part $part -mode out_of_context
if {[llength [get_sites -quiet BUFGCTRL_X0Y16]] != 1} { error "Invalid global clock buffer site" }
if {[get_property HD.CLK_SRC [get_ports clk]] ne "BUFGCTRL_X0Y16"} {
    error "Missing out-of-context global clock source"
}
opt_design
place_design
route_design
report_route_status -file route.rpt
report_timing_summary -delay_type min_max -report_unconstrained -check_timing_verbose -file timing.rpt
report_utilization -file utilisation.rpt
check_timing -verbose -file constraints.rpt
set registers [all_registers]
set result [open internal-timing.tsv w]
puts $result "analysis\tslack_ns\tstart\tend\tdatapath_delay_ns"
foreach delay {max min} {
    set timing [get_timing_paths -from $registers -to $registers -delay_type $delay -max_paths 1 -nworst 1]
    if {[llength $timing] != 1} { error "No register-to-register timing path for $delay" }
    puts $result [join [list $delay [get_property SLACK $timing] [get_property STARTPOINT_PIN $timing] [get_property ENDPOINT_PIN $timing] [get_property DATAPATH_DELAY $timing]] "\t"]
}
close $result
write_verilog -force -mode funcsim routed_function.v
```

Run the implementation in the same directory. The period is set in `timing.xdc`.

```bash
vivado -mode batch -source route.tcl -tclargs xc7z020clg484-1 10 6
```

Check that all routable nets are fully routed in `route.rpt` and review
`constraints.rpt` for missing clocks or unconstrained internal endpoints.
`internal-timing.tsv` reports the worst setup (`max`) and hold (`min`) slack
between two registers. `timing.rpt` also covers the example I/O constraints.
See AMD's [timing summary guidance](https://docs.amd.com/r/2025.2-English/ug906-vivado-design-analysis/Report-Timing-Summary)
for what changes between synthesis and routed timing.

### Bounded square-root lowering

For a positive input code, the highest possible set bit of the radicand after
scaling is the word width plus the fractional width. The `v2` square root
starts at the pair containing that bit and uses the exact bounded integer square
root. Q8.8 therefore starts at bit 24 and needs 13 pairs. The final rounding,
saturation and combinational interface are preserved.

The bounded Q8.8 square-root implementation was rerouted with the same part,
tool build, wrapper and 10 ns clock constraint as the historical row above.
The historical run used six Vivado threads. Pass eight as the last Tcl argument
to reproduce the bounded run.

| Q8.8 square root `v2` | Slice LUTs | Data path (ns) | Setup slack (ns) | Hold slack (ns) |
| --- | ---: | ---: | ---: | ---: |
| Original | 1,693 | 110.230 | -100.332 | 2.199 |
| Bounded | 205 | 24.402 | -14.681 | 0.336 |

The bounded path meets hold and still misses the 100 MHz setup target. The
same change was mapped and checked for equivalence with Yosys 0.70+post in
all six reported fixed-point formats. The other 264 supported standalone
cases emit exactly the same Verilog as the historical comparison.

#### Reuse the trial subtraction

Each square-root digit uses the sign of its bounded trial subtraction to
decide whether to keep the digit and reuses the difference as the next
remainder. The last digit needs only the decision. Scaling, rounding and
the combinational interface are unchanged.

The Q8.8 comparison used Vivado 2025.2 build 6299465, the same
`xc7z020clg484-1` part, eight tool threads, the same clocked wrapper
and the same 10 ns constraint.

| Q8.8 square root `v2` | Slice LUTs | Slice registers | Data path (ns) | Setup slack (ns) | Hold slack (ns) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bounded comparison | 205 | 29 | 24.402 | -14.681 | 0.336 |
| Shared trial subtraction | 144 | 29 | 21.499 | -11.494 | 0.643 |

Both implementations meet hold but miss setup at 100 MHz. The routed
functional netlist agrees with the independent integer reference for all
65,536 input codes. The path and resource figures come from the routed
design reports. Board-level timing and power require separate measurements.
