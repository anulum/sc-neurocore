# Hardware

Hardware abstraction layer for chip emulators and deployment targets.

9 hardware chip emulators: Loihi CUBA, Loihi 2, TrueNorth, BrainScaleS AdEx, SpiNNaker, Akida, DPI, MemristorArray, GenericASIC. Each emulates the target chip's neuron dynamics, precision constraints, and routing limitations.

```python
from sc_neurocore.hardware import LoihiCUBANeuron, TrueNorthNeuron
```

::: sc_neurocore.hardware
    options:
      show_root_heading: true
