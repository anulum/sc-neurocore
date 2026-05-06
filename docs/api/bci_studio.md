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

The primitive layer is the deterministic, auditable closed-loop path for
research/HIL work:

```python
from sc_neurocore.bci_studio.bci_primitives import (
    BCIClosedLoopPrimitive,
    BCIFrame,
    BCIPrimitiveConfig,
)

primitive = BCIClosedLoopPrimitive(
    BCIPrimitiveConfig(
        channels=256,
        sampling_rate_hz=30_000,
        latency_budget_ms=10.0,
        command_threshold_hz=75.0,
    )
)
result = primitive.process_frame(BCIFrame(samples=window, reward=0.0, timestamp_us=1000))
packet = result.feedback_packet
trace = result.trace.as_dict()
```

The trace records schema version, frame id, input shape, spike count, active
channels, score, command, latency, latency-budget status, adaptation status,
and whether the optional native learning bridge was used. Feedback packets are
fixed 24-byte little-endian records suitable for deterministic sink adapters.

Operational boundaries:

- This is research/HIL infrastructure, not medical-device control software.
- Physical feedback requires an explicit sink adapter and external safety case.
- The default command is bounded by `max_feedback_amplitude` and reports when
  clipping was applied.
- `BCIClosedLoopEngine` remains as a compatibility wrapper for older examples.

::: sc_neurocore.bci_studio.bci_primitives

## Studio

::: sc_neurocore.bci_studio.bci_studio
