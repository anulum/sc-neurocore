# Hardware-Aware SNN NAS

NSGA-II evolutionary search over SNN architectures under FPGA resource budgets.

Searches {neuron model, layer width, bitstream length, delay range} jointly —
the first NAS that optimizes hardware parameters alongside topology.

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

## Formal Equivalence

::: sc_neurocore.nas.equiv
    options:
      show_root_heading: true
      members:
        - check_equivalence
        - generate_miter
        - generate_sby
        - EquivResult
