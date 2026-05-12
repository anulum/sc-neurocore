# Sources

Input current sources that drive neurons via SC-encoded bitstreams.

## Bitstream Current Source

Converts multiple scalar inputs through weighted SC synapses into a realised
per-cycle current trace. `step()` consumes that trace one cycle at a time, and
`full_current_estimate()` returns the mean of the same realised trace.

::: sc_neurocore.sources.bitstream_current_source.BitstreamCurrentSource

## Quantum Entropy Source

Optional integration with quantum random number generators
(Qiskit, PennyLane) for true randomness in SC encoding.

::: sc_neurocore.sources.quantum_entropy
