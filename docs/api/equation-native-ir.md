<!--
SPDX-License-Identifier: AGPL-3.0-or-later
Commercial license available
© Concepts 1996–2026 Miroslav Šotek. All rights reserved.
© Code 2020–2026 Miroslav Šotek. All rights reserved.
ORCID: 0009-0009-3560-0851
Contact: www.anulum.li | protoscience@anulum.li
SC-NeuroCore — Equation model compilation and source replay
-->

# Equation models as native IR

`compile_to_ir` compiles an `EquationNeuron` into native integer arithmetic,
comparisons, selections, function programs and state registers. The graph emits
SystemVerilog directly or structured CIRCT HW/Comb/Seq MLIR. Equations and
generated RTL are never executable opaque graph nodes.

The API requires a matching engine build containing word, state, source and
runtime-check instructions. An older installed engine wheel does not gain these
instructions from a Python package update. This API does not automatically
replace a Studio compiler adapter or establish physical FPGA validation.

```python
from sc_neurocore.compiler import compile_to_ir, recompile_ir
from sc_neurocore.neurons.equation_builder import EquationNeuron

model = EquationNeuron(
    {"v": "(I - v + w) / tau", "w": "-w / 8"},
    parameters={"tau": 4.0},
    state={"v": 0.25, "w": -0.5},
    method="rk4",
    dt=0.5,
    threshold="v > 1",
    reset={"v": "0"},
    spike_value="v - w",
)
compiled = compile_to_ir(
    model, "coupled_neuron", data_width=16, fraction=8,
    rounding="nearest", overflow="wrap",
)
system_verilog = compiled.graph.emit_sv()
circt_mlir = compiled.graph.emit_mlir()
saved = compiled.graph.to_text()
restored = recompile_ir(saved)
assert restored.graph.to_text() == saved
assert restored.output_widths == compiled.output_widths
```

The functions are also exported by
`sc_neurocore.compiler.equation_compiler`. Native graphs can be saved and loaded
with the [compute graph text API](sc-ir.md).

## Result and ports

`EquationIR` is an immutable record with these fields:

| Field | Meaning |
|---|---|
| `graph` | Verified native `ScGraph`, including its bound source contract |
| `clocks_per_timestep` | Physical clocks per complete model timestep |
| `pipeline_registers` | Actual selected wide product registers |
| `output_widths` | Ordered `(port_name, physical_width)` tuples |

The hardware clock and asynchronous active-low reset are implicit native ports
`clk` and `rst_n`. `I_t` is the signed raw fixed-point input. State outputs use
`<state>_out`; `spike_out` is one bit. A graded model also emits
`spike_value_out`. Pipelined models expose the maintained `latency` port.
Port-name collisions and native reserved identifiers are errors.

Native scalar state/input ports support 1 through 64 bits. Internal arithmetic
and product words can be wider. A wider requested scalar port raises
`NotImplementedError`; it is not silently narrowed.

## Numerical and timestep semantics

The frontend preserves the existing equation model's five integration methods,
equation declaration order, initial state, parameters, constants, timestep,
threshold, reset order, previous-state aliases and graded expression.
Gauss–Seidel uses the already committed earlier candidates within the same
substep. Compilation does not advance a running model or capture its live
trajectory as a new initial condition.

`signed=True` is required. Product rounding supports `truncate`, `nearest` and
`bankers`. The `overflow` choices are:

| Policy | Integrated state | Reset and graded values |
|---|---|---|
| `saturate` | Clamp at the declared commit boundary | Clamp at the declared event boundary |
| `wrap` | Keep the configured low bits | Keep the configured low bits |
| `trap` | Emit a simulation overflow check before narrowing | Keep the configured low bits |

Trap checks disappear under `SYNTHESIS`. They provide simulation diagnostics,
not synthesized overflow protection.

The default `function_lowering="lut-v1"` is retained; `"v2"` selects the existing
structured function programs. `random_precision` selects `"standard"` or
`"exact"` normal-draw lowering. Random expressions preserve the model's 64-bit
seed, 32-bit stream, call-site numbering and timestep counter.

A positive `pipeline_stages` selects the maintained global multiplication
pipeline policy. Alternatively, `pipeline_points` selects explicit product
names, including names such as `mul0` and `_dt_mul_v`. The two requests are
mutually exclusive. An unmatched explicit name retains the existing no-match
behavior; inspect the actual `pipeline_registers` count.

With product registers, state commits recur every `pipeline_registers + 1`
clocks. Without them, a macro model's `substeps` determine the complete
timestep period. Unsupported pipeline/macro/stochastic combinations refuse
instead of claiming a one-clock update.

## Division domains

Authored variable division needs an encoded nonzero parameter/constant or a
source guard that excludes zero. Quantization matters: a small nonzero source
constant can encode zero and is refused as a variable denominator.
Supported source guards include reversed/chained inequalities and boolean
short circuit, for example `I / v if v > 0 else 0`.

The guard remains part of the lowered model. The frontend does not silently
give every user division a zero result at zero. It also keeps parameter
division distinct from literal reciprocal multiplication: `I / tau` is not
rewritten to `I * (1 / tau)`, whose quantization can differ.

This admission covers proven encoded constants and the supported source
guards. It does not establish arbitrary inferred ranges or correlated
function domains. Totalized divisions inside structured function programs
retain their separate existing semantics.

## Source checks and replay

The graph binds canonical `sc-equation-source.v1` JSON after constructing its
operations. Semantically ordered maps use ordered name/value pairs. The payload
contains equations, initial state, parameters/constants, events, method and
`dt`, detection/substeps, RNG configuration, base/display unit maps, every Q
configuration field, requested pipeline selection and commit-policy labels.
Live state and live random counters are excluded from compilation inputs.

`graph.source_payload`, `source_sha256` and `source_graph_sha256` expose the
exact payload and its native checksums. Native text parsing checks both source
bytes and the canonical graph binding. Checksums do not authenticate an author
or establish that an untrusted description matches a graph's behavior.

`recompile_ir(text)` additionally reconstructs the original model and numerical
configuration, recompiles it, and requires exact canonical graph-text equality.
Strict-unit models repeat dimensional validation using their base values and
retain their separate display units. Display units must name exactly the declared
states and have dimensions compatible with those states; a valid source checksum
does not make an unknown state or an incompatible unit admissible. Malformed, extra, duplicate or
noncanonical source fields refuse. A validly rehashed different equation still
fails if it does not reproduce the graph.

Legacy graphs without equation source metadata remain readable with
`parse_ir`; they cannot be replayed as equation models.
