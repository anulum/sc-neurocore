# Sigma-delta source-fidelity receipt

Model 51 resolves an identity conflict. The old bipolar accumulator did not
contain the filtered signal, local reconstruction, or reconstruction filter
claimed by its Yoon attribution. The closure therefore keeps both models:

- `SigmaDeltaNeuron`: sampled APSDM specialization of WO2016022241A1 equations
  20-27 and 40, with unipolar output.
- `SCSigmaDeltaAccumulatorNeuron`: exact retained signed project recurrence.

The source receipt is an independent transcription, not a call through a
production backend. It uses 512 inputs, records 276 events, and hashes
little-endian `(sigma, reconstruction, event)` rows to
`4c22a86d4fe810dad3d5a039717798c5183ff03e9f00d8317e5568a5ad5b8d99`.
The SC receipt separately pins the former code source hash and its original
256-step binary64 trace.

Five runtime dispatchers, paired schemas, source/binary-bound benchmarks, and
two bounded fixed-point units prevent the identities from silently converging.
The source model is clocked; continuous-valued event instants, biological
fitting, network effects, device validation, timing, and PPA remain outside the
claim.
