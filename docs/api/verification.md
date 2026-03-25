# Verification

Formal and functional verification utilities for SNN designs.

- Temporal property checking: verify that SNN outputs satisfy temporal logic specifications
- Equivalence checking: verify Python simulation matches Verilog RTL bit-for-bit
- Coverage metrics: track which neuron states and transitions have been exercised

7 SymbiYosys formal verification scripts + 67 properties in `hdl/formal/`.

```python
from sc_neurocore.verification import TemporalPropertyChecker
```

::: sc_neurocore.verification
    options:
      show_root_heading: true
