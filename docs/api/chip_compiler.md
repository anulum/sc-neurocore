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
        - BUILTIN_CHIPS

## Compiler

::: sc_neurocore.chip_compiler.compiler
    options:
      show_root_heading: true
      members:
        - compile_for_chip
        - CompilationResult
        - CoreMapping
