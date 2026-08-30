# SCSymmetricQuadraticIFNeuron

**Module:** `sc_neurocore.neurons.models.quadratic_if`

This count-neutral project identity preserves SC-NeuroCore's historical
quadratic integrate-and-fire profile:

- `v=-1`
- `v_reset=-1`
- `v_peak=+1`
- `dt=0.01`
- exact held-current Riccati flow in the production runtimes
- inclusive finite event boundary at `v_peak`

It remains available through both `QuadraticIFNeuron()` and the explicit
`SCSymmetricQuadraticIFNeuron` class. Its paired
`sc_symmetric_quadratic_if.toml`/JSON schemas, independent zero-current trace,
all five complete runtime lanes, and the original `sc_quadratic_if` RTL/formal
lane preserve existing experiments without attributing these finite boundaries
to Latham et al. (2000). It does not add to the literature model count.

For the source-counted identity, Latham's normalized `31/3` apex, `-3` reset,
`.05` timestep, DOI receipt, controlled benchmark, and dedicated source RTL,
see [QuadraticIFNeuron](quadratic_if.md).
