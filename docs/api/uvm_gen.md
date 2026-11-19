# UVM Generator

The UVM generator turns a supported Verilog/SystemVerilog module header into a
deterministic bundle of transaction, sequence, driver, monitor, scoreboard,
coverage, agent, environment, top-level, assertion-bind, SymbiYosys,
simulator-Makefile, and regression-list text.

The historical `sc_neurocore.uvm_gen.uvm_gen` module is a stable compatibility
facade. Focused private modules now own RTL parsing, configuration contracts,
generated-artifact packaging, UVM component emission, simulator/formal harness
emission, and orchestration. The package-level 13-name export set, callable
signatures, class identities, and pickle-qualified paths remain compatible with
the pre-split implementation while `RTLModule.from_verilog_source` evaluates
conditional compilation directives (`ifdef`, `ifndef`, `elsif`, `else`,
`endif`) against an optional `defines` set (empty synthesis view by default),
resolves parameterised packed and unpacked dimensions (grouped `parameter` and
body `localparam` constant expressions including `$clog2`, IEEE 1800
left-associative `**` below unary operators, exact integer `/` and `%`
truncation, bitwise/shift operators, `signed`/`unsigned` qualifiers with or
without separating whitespace before `[`, and ternary selectors bounded within
`2**64`), ANSI port initialisers, non-ANSI Verilog port lists, AMBA AXI
clock/reset conventions, and sized bounds wider than 32 bits, failing closed on
unrecognised or interface port items, constant function calls, and unsupported
compiler directives (`define`, `include`, and macro invocations) inside the
parsed parameter and port declarations.

The focused UVM cohort contains 90 tests. All 774 statements and 278 branches
across the eight governed package modules are covered, including six pinned
deterministic bundle SHA-256 digests for default and configured LIF, dense, and
port-less inputs plus a Git-tracked repository RTL header corpus check. Strict
typing and NumPy-docstring policy cover the complete split surface.

!!! warning "Draft verification-IP surface"
    The repository architecture map continues to classify `uvm_gen` as a draft
    stub surface. This modularisation proves Python API and deterministic-text
    compatibility; it does not claim that every emitted bundle compiles under
    a vendor UVM installation or that the generated SymbiYosys harness is
    promotion-ready. The auto-generated Go, Rust, Julia, and Mojo placeholder
    mirrors are not wired runtime counterparts and remain separate backlog
    debt. No cross-language or throughput claim applies to this static text
    generator.

## Quick Start

```python
from sc_neurocore.uvm_gen import (
    UVMGenerator, RTLModule, StimulusConfig, CoverageSpec,
)
```

::: sc_neurocore.uvm_gen.uvm_gen
