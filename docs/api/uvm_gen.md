# UVM Generator

UVM testbench generator for SC neuromorphic IP verification.
Emits constrained-random stimulus, self-checking scoreboards,
functional coverage, and formal property harnesses.

## Quick Start

```python
from sc_neurocore.uvm_gen.uvm_gen import (
    UVMGenerator, RTLModule, StimulusConfig, CoverageSpec,
)
```

::: sc_neurocore.uvm_gen.uvm_gen
