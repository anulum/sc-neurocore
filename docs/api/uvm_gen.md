# UVM Generator

UVM testbench generator for SC neuromorphic IP verification.
Emits constrained-random stimulus, self-checking scoreboards,
functional coverage, and formal property harnesses.

The public generator module is covered by the scoped NumPy-docstring policy.
`tests/test_uvm_gen/test_uvm_gen.py` also exercises the parameter-less module
parser branch and blank port-entry handling, and the current isolated coverage
run reports 100% for `src/sc_neurocore/uvm_gen/uvm_gen.py`.

## Quick Start

```python
from sc_neurocore.uvm_gen.uvm_gen import (
    UVMGenerator, RTLModule, StimulusConfig, CoverageSpec,
)
```

::: sc_neurocore.uvm_gen.uvm_gen
