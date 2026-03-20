# Compiler

Network-to-hardware compilation pipeline.

## Equation → Verilog Compiler

Compile arbitrary ODE neuron equations to synthesizable Verilog RTL.
The only framework that goes from a string equation to FPGA hardware.

::: sc_neurocore.compiler.equation_compiler

## Pipeline

::: sc_neurocore.compiler.pipeline

## MLIR Emitter

::: sc_neurocore.compiler.mlir_emitter

## Weight Quantizer

Float → Q-format fixed-point with nearest/stochastic/floor rounding,
plus SC probability mapping.

::: sc_neurocore.compiler.quantizer

## IR Type Checker

Validates Stochastic IR graphs before emission. Catches Bitstream/Rate/Spike
type mismatches that would otherwise silently produce wrong results.

::: sc_neurocore.compiler.ir_type_checker
