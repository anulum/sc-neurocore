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

`DVSInputLayer` validates sensor dimensions, decay constants, AER event
addresses, event timestamps, polarities, and generated bitstream lengths before
mutating the event-density surface. Rejected event batches leave
`surface` and `last_update_time` unchanged. Empty batches return the current
probability frame without exposing mutable internal state.

Event coordinates outside the configured frame are ignored as sparse
out-of-field sensor noise, but malformed coordinate types are rejected. Valid
events are integrated in timestamp order, decayed by `decay_tau`, converted
through `tanh(surface)`, and exposed as `[0, 1]` probabilities for stochastic
bitstream generation.

The Rust safety mirror (`accel/rust/safety/dvs_input.rs`), Julia validation
mirror (`accel/julia/interfaces/dvs_input.jl`), and Mojo FFI validation shim
(`accel/mojo/kernels/dvs_input.mojo`) enforce the same geometry, decay,
timestamp, polarity, and bitstream-length boundaries. The Python benchmark in
`benchmarks/benchmark_advanced_modules.py` uses monotonic precomputed DVS event
batches so the measured path respects the same layer-clock contract.

## Real World

::: sc_neurocore.interfaces.real_world
