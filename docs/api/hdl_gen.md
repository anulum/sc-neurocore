# HDL Generation

Verilog RTL generation from Python SNN descriptions.

- `generate_verilog()` — Convert SC layer/neuron descriptions to synthesizable Verilog modules. Supports Q8.8 fixed-point, LFSR encoders, popcount trees, and event-driven AER.
- IR compiler pipeline: Python → intermediate representation → SystemVerilog / MLIR (CIRCT backend)

19 hand-written Verilog modules + equation-to-Verilog compiler for arbitrary ODEs.

```python
from sc_neurocore.hdl_gen import generate_verilog
```

::: sc_neurocore.hdl_gen
    options:
      show_root_heading: true
