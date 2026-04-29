# Interfaces

External I/O bridges: brain-computer interface protocols, CCW audio bridge,
dynamic vision sensor input, and real-world actuator output.

## BCI

::: sc_neurocore.interfaces.bci

## Closed-Loop BCI HIL

::: sc_neurocore.interfaces.bci_closed_loop

::: sc_neurocore.interfaces.bci_hil_manifest

`build_bci_hil_reference_manifest()` exposes deterministic reference
manifests for `pynq_shd` and `probe_384ch`. Both use the
`ClosedLoopBCITemplate` path:

```text
raw waveform window
  -> WaveformCodec compression
  -> threshold spike raster
  -> AER payload generation
  -> rate decoder
  -> feedback frame
  -> DeviceTelemetry summary
```

The `pynq_shd` profile uses the repository's documented SHD topology
(`700 -> 256 -> 20`) as the reference model shape and defaults to the
in-process implant emulator. Physical PYNQ use still requires the external
bitstream and an explicit hardware feedback sink.

## CCW Bridge

::: sc_neurocore.interfaces.ccw_bridge

## DVS Input

::: sc_neurocore.interfaces.dvs_input

## Real World

::: sc_neurocore.interfaces.real_world
