# Hardware-Aware SNN NAS

NSGA-II evolutionary search over SNN architectures under FPGA resource budgets.

Searches {neuron model, layer width, bitstream length, delay range} jointly —
the first NAS that optimizes hardware parameters alongside topology.

The public surface exposes derived architecture helpers for layer count,
compiler-facing layer dimensions, and dense connection counts. Search and
equivalence result objects also provide compact textual summaries for logs,
reports, and CI artifacts.

## Search Space

::: sc_neurocore.nas.search_space
    options:
      show_root_heading: true
      members:
        - Architecture
        - SearchSpace

## Search Engine

::: sc_neurocore.nas.search
    options:
      show_root_heading: true
      members:
        - nas
        - NASResult

## Hardware-Aware SC-NAS Engine

`sc_neurocore.nas.sc_nas_engine` provides the evolutionary SC-NAS surface used
for bitstream-length, decorrelator, neuron-family, and FPGA-resource search.
It evaluates candidate resource estimates, extracts a Pareto front, emits
SystemVerilog parameter shells for selected candidates, and can route
tournament selection through the optional Rust extension when that extension is
available at import time.

::: sc_neurocore.nas.sc_nas_engine
    options:
      show_root_heading: true
      members:
        - DecorrelationStrategy
        - NeuronType
        - FPGAResourceBudget
        - NASObjective
        - LayerConfig
        - SCCandidate
        - SCFitnessEvaluator
        - pareto_front
        - EvolutionaryNAS
        - NASReport
        - run_nas
        - NASVerilogEmitter

## Differentiable SC-NAS

`sc_neurocore.nas.darts_sc_nas` provides the DARTS relaxation used to train
bitstream-length choices through Gumbel-Softmax architecture weights. Its public
surface documents candidate variance injection, mixed-operation resource costs,
optimal bitstream extraction, and network-level hardware penalties.

::: sc_neurocore.nas.darts_sc_nas
    options:
      show_root_heading: true
      members:
        - BitstreamCandidate
        - SCMixedOp
        - SCNASNetwork

## Formal Equivalence

::: sc_neurocore.nas.equiv
    options:
      show_root_heading: true
      members:
        - check_equivalence
        - generate_miter
        - generate_sby
        - EquivResult
