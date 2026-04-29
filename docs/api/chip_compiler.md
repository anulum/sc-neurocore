# Multi-Chip Hardware Compiler

Target-agnostic SNN compiler with built-in specs for Loihi 2, SynSense Xylo/Speck,
BrainChip Akida, SpiNNaker2, and BrainScaleS-2.

## Chip Specifications

::: sc_neurocore.chip_compiler.chip_spec
    options:
      show_root_heading: true
      members:
        - ChipSpec
        - CoreSpec
        - load_chip_spec
        - BUILTIN_CHIPS

`load_chip_spec()` accepts UTF-8 JSON files for custom chip targets. The
loader rejects non-object roots, missing or unexpected fields, invalid scalar
types, non-finite numeric values, unsupported routing topologies, and malformed
core specifications before constructing `ChipSpec` / `CoreSpec` objects.

## Compiler

::: sc_neurocore.chip_compiler.compiler
    options:
      show_root_heading: true
      members:
        - compile_for_chip
        - CompilationResult
        - CoreMapping
