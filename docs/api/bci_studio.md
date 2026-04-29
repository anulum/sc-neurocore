# BCI Studio

Brain-computer interface closed-loop control. Real-time neural
decoding, closed-loop stimulation, and charge density safety limits.

## Quick Start

```python
from sc_neurocore.bci_studio.bci_primitives import BCIClosedLoopEngine
from sc_neurocore.bci_studio.bci_studio import BCIStudio
from sc_neurocore.interfaces import (
    build_bci_hil_reference_manifest,
    create_bci_hil_template,
)
```

## HIL Reference Path

For deterministic hardware-in-the-loop prototyping, use the interface-layer
template rather than the legacy studio loop:

```python
manifest = build_bci_hil_reference_manifest("pynq_shd")
template = create_bci_hil_template("pynq_shd")
```

The reference path wires raw probe-like waveform windows through
`WaveformCodec`, `AERSpikeCodec`, rate decoding, feedback emission, and
`DeviceTelemetry`. The default sink is an implant emulator; physical PYNQ
feedback requires an explicit sink adapter and external bitstream artefacts.

## Primitives

::: sc_neurocore.bci_studio.bci_primitives

## Studio

::: sc_neurocore.bci_studio.bci_studio
